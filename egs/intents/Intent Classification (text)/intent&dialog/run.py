import asyncio
import tornado.ioloop
import inspect
import time
from speechcloud.sc_client import SpeechCloudClient


class SpeechCloudKit(SpeechCloudClient):

    def __init__(self, speechcloud_url):
        SpeechCloudClient.__init__(self, speechcloud_url)
        self.session_uri = None

    def _recognizing(self):
        print('Recognizing')

    def _paused(self):
        print('Paused')

    def _ready(self, **kwargs):
        print('Ready')

    def _result(self, **kwargs):
        print('Result')

    def _log_audio_record(self, uri, **kwargs):
        print("ASR input stored as: %s", uri)

    async def run(self, dialog_class):
        source_fn = inspect.getfile(dialog_class)
        print("Loaded dialog class %s from %s", dialog_class.__name__, source_fn)
        self.on("asr_recognizing", self._recognizing)
        self.on("asr_paused", self._paused)
        self.on("asr_ready", self._ready)
        self.on("asr_result", self._result)
        self.on("asr_audio_record", self._log_audio_record)
        self.on("sc_start_session", self._start_session)

        print("Waiting for button press (2 sec)")
        time.sleep(2)

        kwargs = {}
        await self.run_async(dialog_class, **kwargs)

    def _start_session(self, session_uri, **kwargs):
        self.session_uri = session_uri

    def log_asr_ready(self, **kwargs):
        print("Received asr_ready event")
            
async def main():
    URI = 'https://cak.zcu.cz:9443/v1/speechcloud/edu/bp/jivl'
    sc = SpeechCloudKit(URI)
    await sc.run(MyDialog)
    
if __name__ == "__main__":

    from my_dialog import MyDialog
    
    tornado.ioloop.IOLoop.current().run_sync(main)
