import random
from speechcloud.dialog import Dialog
from speechcloud.quick_answer import search_dialog


class MyDialog(Dialog):

    async def main(self):
        voices = ["Jan210", "Stanislav210", "Jiri210"]
        HLAS = random.choice(voices)
        await search_dialog.search_dialog(self, HLAS)
