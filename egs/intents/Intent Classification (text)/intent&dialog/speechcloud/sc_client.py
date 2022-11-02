from __future__ import print_function
from event_emitter.events import EventEmitter
import json
import pjsua2 as pj
import time
from pprint import pprint
import requests
import logging
from urllib.parse import urlparse
import threading
import queue
from speechcloud.dialog import SpeechCloudWS
from tornado.websocket import websocket_connect
from tornado.ioloop import IOLoop
import subprocess

logger = logging.getLogger('dialog_sip')


class SpeechCloudAccount(pj.Account):
    registered = False

    def onRegState(self, prm):
        if prm.code == pj.PJSIP_SC_OK:
            self.registered = True
            logger.info("SpeechCloud registration OK")


class SpeechCloudError(RuntimeError):
    pass


class SpeechCloudCall(pj.Call):
    def __init__(self, speechcloud_ws, account, adev):
        super(SpeechCloudCall, self).__init__(account)
        self.speechcloud_ws = speechcloud_ws
        self.adev = adev
        # Flag indicating that the call was confirmed
        self.was_confirmed = False

    def onCallState(self, prm):
        ci = self.getInfo()
        logger.debug("SpeechCloudCall: %s [%s, %s]", ci.remoteUri, ci.stateText, ci.state)

        if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            logger.debug("Call disconnected, closing SpeechCloud WS")
            if not self.was_confirmed:
                # The SIP call was directly disconnected without confirming the call
                self.speechcloud_ws.error_pjsip_call("SpeechCloud disconnected the SIP call")

        elif ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            logger.debug("Call confirmed, bridging with sound device")
            self.was_confirmed = True

            for idx, media in enumerate(ci.media):
                if media.type == pj.PJMEDIA_TYPE_AUDIO:
                    stream_info = self.getStreamInfo(idx)

                    logger.info("Call codec: %s/%s, remote RTP address: %s", stream_info.codecName, stream_info.codecClockRate, stream_info.remoteRtpAddress)

                    aud_med = self.getAudioMedia(idx)
                    aud_med.startTransmit(self.adev.getPlaybackDevMedia())
                    self.adev.getCaptureDevMedia().startTransmit(aud_med)
                    

class SpeechCloudClient(SpeechCloudWS):
    PJ_MAKE_CALL = "make_call"
    PJ_TERMINATE = "terminate"

    CONNECT_TIMEOUT = 10
    UPDATE_STATS_PERIOD = 1

    logger = logging.getLogger("SpeechCloudClient")

    def __init__(self, app_uri, intro_wav=None):
        EventEmitter.__init__(self)
        self.ws_conn = None
        self.ioloop = None
        self.app_uri = app_uri
        self.intro_wav = intro_wav
        self.pj_registration_event = threading.Event()
        # Raise this error at the end of the websocket_main() as the result of not connected SIP call
        self.pjsip_error = []
        self.on("sc_start_session", self.init_pjsip_call)
        self.on("asr_ready", self.log_asr_ready)

    async def run_async(self, dialog_class, verbose=False, intro_wav=None):
        self.ioloop = IOLoop.current()
        self.account_config, self.client_wss = self.get_speechcloud_account_config(self.app_uri)

        if intro_wav is None:
            intro_wav = self.intro_wav

        if intro_wav:
            self.intro_thread = threading.Thread(target=self.intro_main, args=(intro_wav,), daemon=True)
            self.intro_thread.start()
        else:
            self.intro_thread = None

        self.pj_cmd_queue = queue.Queue()
        self.pj_registration_event.clear()
        self.pj_registration_state = False
        self.pjsua_thread = threading.Thread(target=self.pjsua_main, kwargs=dict(verbose=verbose), daemon=True)
        self.pjsua_thread.start()

        try:
            self.pj_registration_event.wait(timeout=self.CONNECT_TIMEOUT)
            if self.pj_registration_state:
                self.logger.info("SpeechCloud SIP registered, will connect to %s", self.client_wss)
                # Initialize the dialog class
                self.initialize(dialog_class)
                # And run the main WebSocket hangling loop
                return await self.websocket_main()
            else:
                self.logger.error("Cannot register to SpeechCloud SIP")
                raise SpeechCloudError("Cannot register to SpeechCloud SIP")
        finally:
            self.pj_cmd_queue.put(self.PJ_TERMINATE)
            self.logger.debug("Joining pjsua_thread")
            self.pjsua_thread.join()
            self.ioloop = None

    def run(self, dialog_class, verbose=False):
        async def main():
            return await self.run_async(dialog_class, verbose)

        try:
            IOLoop.current().run_sync(main)
        finally:
            self.logger.debug("Emitting sc_destroyed for session: %s", self.dm.session_id)
            self.emit("sc_destroyed", session_id=self.dm.session_id)
            self.close()

    def pjsua_call_stats(self, call):
        call_info = call.getInfo()

        if len(call_info.media) != 1:
            self.logger.warning("Call info has not exactly 1 media")

        call_stats = call.getStreamStat(0)
        self.rtt_delay = (call_stats.rtcp.rttUsec.mean / 1000 + call_stats.jbuf.avgDelayMsec) / 1000


    def pjsua_main(self, verbose=False):
        self.logger.debug("Initializing PJSUA2 library (starting thread)")
        # Create and initialize the library
        ep_cfg = pj.EpConfig()

        ep_cfg.uaConfig.maxCalls = 1
        ep_cfg.uaConfig.threadCnt = 0
        if not verbose:
            # Lower log verbosity 
            ep_cfg.logConfig.consoleLevel = 1

        # Set clockrate to 48kHz to support OPUS without resampling
        ep_cfg.medConfig.clockRate = 48000
        ep_cfg.medConfig.sndClockRate = 48000

        # Configure jitter-buffer
        ep_cfg.medConfig.jbInit = 200
        ep_cfg.medConfig.jbMinPre = 100
        ep_cfg.medConfig.jbMaxPre = 300

        ep = pj.Endpoint()
        ep.libCreate()
        ep.libInit(ep_cfg)

        # Create SIP transport. Error handling sample is shown
        sipTpConfig = pj.TransportConfig()
        #sipTpConfig.port = self.SIP_PORT
        ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, sipTpConfig)
        # Start the library
        ep.libStart()

        self.logger.debug("PJSUA2 library started")

        adev = ep.audDevManager()
        # Disable Echo Cancelation
        adev.setEcOptions(0, 0)

        # Set codec priorities (use wide-band OPUS codec)
        ep.codecSetPriority("opus/48000", 150)

        account = call = None
        try:
            account = SpeechCloudAccount()
            account.create(self.account_config)

            self.logger.debug("Waiting for SpeechCloud SIP registration")

            t0 = time.monotonic()
            # Wait until the account is registered
            while not account.registered:
                ep.libHandleEvents(10)
                if time.monotonic() - t0 > self.CONNECT_TIMEOUT:
                    # Notify the main thread about the change in registration
                    self.pj_registration_event.set()
                    self.logger.critical("Cannot register for SpeechCloud SIP within %d seconds", self.CONNECT_TIMEOUT)
                    raise SpeechCloudError("Cannot register for SpeechCloud SIP within {} seconds".format(self.CONNECT_TIMEOUT))

            # Signal the registered event
            self.pj_registration_state = True
            self.pj_registration_event.set()

            # Run the main event loop
            self.logger.debug("Running PJSUA2 thread main loop")

            # Time when the SIP call was initiated
            t_call = None
            t_stats = None
            while True:
                ep.libHandleEvents(10)
                try:
                    cmd = self.pj_cmd_queue.get(False)
                    if cmd == self.PJ_MAKE_CALL:
                        if call is not None:
                            self.logger.critical("Received multiple PJ_MAKE_CALL")
                            raise SpeechCloudError("Received multiple PJ_MAKE_CALL")

                        ep.libHandleEvents(10)

                        # Set the beginning of the SIP call
                        t_call = time.monotonic()
                        t_stats = t_call
                        call = SpeechCloudCall(self, account, adev)
                        self.logger.debug("Starting SIP call to SpeechCloud, timer started")
                        prm = pj.CallOpParam()
                        call.makeCall("sip:anything", prm)
                    elif cmd == self.PJ_TERMINATE:
                        self.logger.debug("Terminating PJSUA2 thread")
                        break
                except queue.Empty:
                    pass

                if t_call:
                    # Check for SIP call timeouts
                    if call.was_confirmed:
                        self.logger.debug("SIP call was made successfully, canceling timer")
                        t_call = None
                    elif time.monotonic() - t_call > self.CONNECT_TIMEOUT:
                        self.logger.critical("SIP call cannot be made within %d seconds", self.CONNECT_TIMEOUT)
                        raise SpeechCloudError("SIP call cannot be made within {} seconds".format(self.CONNECT_TIMEOUT))

                if t_stats:
                    # Update the RTCP stats
                    if time.monotonic() - t_stats > self.UPDATE_STATS_PERIOD:
                        self.pjsua_call_stats(call)
                        t_stats = time.monotonic()
        except Exception as e:
            # Store the exception and raise it at the end of websocket_main
            self.error_pjsip_call(e)
        finally:
            ep.hangupAllCalls()
            account.shutdown()

            # Wait 0.1 second to process all messages
            for i in range(10):
                ep.libHandleEvents(10)

            del call
            del account

            ep.libDestroy()

            self.logger.debug("PJSUA2 library destroyed (thread stopped)")

    def get_speechcloud_account_config(self, app_uri):
        response = requests.get(app_uri)
        config = response.json()

        acfg = pj.AccountConfig()
        acfg.callConfig.timerMinSESec = 90
        acfg.callConfig.timerSessExpiresSec = 1800

        # Get domain name from sip_uri
        realm = urlparse(config["sip_uri"].replace(":", "://", 1)).hostname

        acfg.idUri = "sip:"+config["sip_username"]
        acfg.regConfig.registrarUri = config["sip_uri"]
        cred = pj.AuthCredInfo("digest", realm, config["client_id"], 0, config["sip_password"])
        acfg.sipConfig.authCreds.append(cred)
        acfg.sipConfig.proxies.append(config["sip_uri"])
        return acfg, config["client_wss"]

    async def write_message(self, msg):
        if self.ws_conn is None:
            raise ValueError("Cannot write_message, because WebSocket connection is not open")

        msg = json.dumps(msg)
        await self.ws_conn.write_message(msg)

    async def websocket_main(self):
        # We want to activate the client to send the schema
        try:
            try:
                self.ws_conn = await websocket_connect(self.client_wss+"?activate_client=1", connect_timeout=self.CONNECT_TIMEOUT)
            except Exception as e:
                self.logger.exception("Cannot connect to SpeechCloud WebSocket")
                raise SpeechCloudError("Cannot connect to SpeechCloud WebSocket") from e

            self.logger.debug("WebSocket connected, calling self.open")

            self.open()
            while True:
                msg = await self.ws_conn.read_message()
                if msg is not None:
                    await self.on_message(msg)
                else:
                    # The connection is closed, break the cycle
                    self.on_close()
                    break

            if self.pjsip_error:
                # We have accumulated the PJSIP error, so raise the first one
                raise self.pjsip_error[0]
        finally:
            self.ws_conn = None

    def close(self):
        if self.ws_conn:
            return self.ws_conn.close()
        else:
            self.logger.info("close() called, but no WebSocket connection, ignoring")

    def made_pjsip_call(self):
        self.logger.info("SIP call connected")
        self.successful_call = True

    def error_pjsip_call(self, msg):
        # this signalizes the error while creating the SIP call
        if self.ioloop is None:
            raise SpeechCloudError("SpeechCloudWS.error_pjsip_call() called while the main ioloop is inactive")
        else:
            # Schedule WebSocket close
            self.successful_call = False
            if isinstance(msg, str):
                self.pjsip_error.append(SpeechCloudError(msg))
            else:
                self.pjsip_error.append(msg)
            self.ioloop.add_callback(self.close)

    def intro_main(self, intro_wav):
        try:
            self.logger.debug("Playing intro WAV: %s", intro_wav)
            # To remove some glitches on RPi wait a small amount of time
            time.sleep(0.1)
            self.intro_popen = subprocess.Popen(["play", intro_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.intro_popen.wait()
            self.logger.debug("Intro WAV finished")
        finally:
            self.intro_thread = None
            del self.intro_popen

    def intro_join(self):
        try:
            self.intro_thread.join()
            self.intro_thread = None
        except AttributeError:
            # self.intro_thread is already None
            pass

    @property
    def intro_running(self):
        return self.intro_thread is not None and self.intro_popen.poll() is None

    def intro_cancel(self):
        self.logger.info("Canceling intro WAV")
        try:
            self.intro_popen.kill()
        except AttributeError:
            pass

    def init_pjsip_call(self, **kwargs):
        self.logger.debug("Scheduling PJ_MAKE_CALL")
        self.pj_cmd_queue.put(self.PJ_MAKE_CALL)

    def log_asr_ready(self, **kwargs):
        self.logger.info("Received asr_ready event")
        if self.intro_thread is not None:
            self.logger.debug("Waiting for intro WAV to finish")
            self.intro_join()

    def log_dialog_exception(self, exc_info):
        self.logger.error("Error in a dialog class", exc_info=exc_info)
        # Store the exception and reraise it at the and of websocket_main()
        self.pjsip_error.append(exc_info[1])