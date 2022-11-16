import random
import asyncio
import logging
from speechcloud.quick_answer import google_it
import json


async def search_dialog(self, HLAS):

    # load phrases.json
    with open('speechcloud/quick_answer/phrases.json', 'r', encoding='utf-8') as f:
        phrases = json.load(f)

    await self.synthesize_and_wait(text="Dobrý den! Jsem děda Vševěda.", voice=HLAS)
    while True:
        input('Next session?')

        text = random.choice(phrases['opening'])
        result = await self.synthesize_and_wait_for_asr_result(text=text, voice=HLAS, timeout=20)

        while result is None:
            logging.info("Žádný výsledek nerozpoznán")
            text = random.choice(phrases['no_result'])
            result = await self.synthesize_and_wait_for_asr_result(text=text, voice=HLAS, timeout=20)

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
