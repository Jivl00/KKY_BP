import random
from dialog import Dialog
import asyncio
import logging
import google_it
import json


class MyDialog(Dialog):

    async def main(self):
        voices = ["Jan210", "Stanislav210", "Jiri210"]
        HLAS = random.choice(voices)

        # load phrases.json
        with open('phrases.json', 'r', encoding='utf-8') as f:
            phrases = json.load(f)

        await self.synthesize_and_wait(text="Dobrý den! Jsem děda Vševěda.", voice=HLAS)
        while True:
            input('Next session?')

            text = random.choice(phrases['opening'])
            result = await self.synthesize_and_wait_for_asr_result(text=text, voice=HLAS, timeout=10)

            while result is None:
                logging.info("Žádný výsledek nerozpoznán")
                text = random.choice(phrases['no_result'])
                result = await self.synthesize_and_wait_for_asr_result(text=text,
                                                                       voice=HLAS, timeout=10)

            asr_res = result["result"]
            print('Waiting...')
            print(asr_res)
            text = random.choice(phrases['thinking'])
            await self.synthesize_and_wait(text=text, voice=HLAS)
            google_it_res = google_it.search(asr_res)
            print('Synthesizing...')
            text = random.choice(phrases['answer'])
            if google_it_res == '':
                google_it_res = random.choice(phrases['dont know'])
            await self.synthesize_and_wait(text=f"{text} {google_it_res}", voice=HLAS)
