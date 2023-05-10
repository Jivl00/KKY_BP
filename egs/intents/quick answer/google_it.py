'''
NOTE: Google constantly keeps on changing its SERP structure and overall algorithm
so this code will eventually stop working
'''

import re
import time

import requests
from bs4 import BeautifulSoup


def search(google_question):
    if 'pocasi' in google_question:
        import seznam_cz
        return seznam_cz.search(google_question)
    # print(google_question)
    google_question = re.sub(r'[^\w\s]', '', (google_question.lower())).replace(' ', '%20')
    url = 'https://www.google.com/search?q=' + google_question
    # print(url)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/64.0.3282.186 Safari/537.36'}
    r = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(r.text, 'lxml')

    for result in (soup.find('div', class_='sXLaOe'), soup.find('div', class_='ayqGOc kno-fb-ctx KBXm4e'),
                   soup.find('div', class_='Z0LcW t2b5Cf'), soup.find('div', class_='dDoNo ikb4Bb gsrt'),
                   soup.find('div', class_='wob_dcp'),
                   soup.find('div', class_='ayqGOc kno-fb-ctx kpd-lv kpd-le KBXm4e'),
                   soup.find('div', class_='zCubwf'), soup.find('div', class_='kp-hc'), soup.find('a', class_='FLP8od'),
                   soup.find('span', class_='ILfuVd'), soup.find('div', class_='LGOjhe'),
                   soup.find('div', class_='gsrt vk_bk FzvWSb YwPhnf'),
                   soup.find('div', class_='dDoNo vrBOv vk_bk'), soup.find('ul', class_='i8Z77e'),
                   soup.find('div', class_='dbg0pd'), soup.find('ol', class_='X5LH0c'),
                   soup.find('div', class_='oSioSc').find('span', class_='Y2IQFc')
                   if soup.find('div', class_='oSioSc') is not None else None,
                   soup.find('tr', class_='ztXv9').find('th')
                   if soup.find('tr', class_='ztXv9') is not None else None):
        if result is not None:
            print(result.text)
            return result.text
    print('No answer found.')
    return 'Neznám odpověď. Polož jednodušší otázku, prosím.'

#
# google_question = 'Kdy se narodil Trump?'
# search(google_question)
# google_question = 'Kdy se narodila kralovna Alzbeta?'
# search(google_question)
# google_question = 'Jake je datum Narozeni kralovny Alzbety'
# search(google_question)
google_question = 'Hlavní město Kanady'
print(search(google_question))
# google_question = 'Kdy zacina podzim'
# search(google_question)
# google_question = 'Kdy zacina jaro'
# search(google_question)
# google_question = 'Kdy zacina leto'
# search(google_question)
# google_question = 'Pocet obyvatel Plzne'
# search(google_question)
# google_question = 'jake je hlavní město cech'
# search(google_question)
# google_question = 'hlavní město makedonie'
# search(google_question)
google_question = 'pocasi plzen'
search(google_question)
google_question = 'Jake je pocasi v Praze?'
search(google_question)
# google_question = 'jaky je kurz eura'
# search(google_question)
google_question = 'jake je nejvetsi zvire'
search(google_question)
google_question = 'pocet kosti v lidskem tele'
search(google_question)
google_question = 'jaky je vzorec vody'
search(google_question)
google_question = 'kolik je 100 kč na eura'
search(google_question)
google_question = 'kolik je hodin'
search(google_question)
google_question = 'jaky je aktualni cas'
search(google_question)
google_question = 'Co to je recese'
search(google_question)
google_question = 'Jake jsou akcie Netflix'
search(google_question)
google_question = 'Pocet obyvatel Evropy'
search(google_question)
google_question = 'Jaka je populace Londyna'
search(google_question)
google_question = '100 cm na metry'
search(google_question)
google_question = 'Kolik je 100 cm na metry'
search(google_question)
google_question = 'fakulta aplikovaných věd adrasa'
search(google_question)
google_question = 'linkin park nejlepsi skladby'
search(google_question)
# google_question = 'jak se jmenuje zpevak Bring me the Horizont'
# search(google_question)
# google_question = 'jaka je rozloha dubaje'
# search(google_question)
# google_question = 'jak stary je vesmir'
# search(google_question)
# google_question = 'kolik penez ma Elon Musk'
# search(google_question)
# google_question = 'jake je cislo na policii'
# search(google_question)
# google_question = 'Jak rychle zhubnout'
# search(google_question)
# google_question = 'pes překlad anglicky'
# search(google_question)
google_question = 'anglicky preklad žralok'
search(google_question)
# google_question = 'Jak se rekne nemecky žralok'
# search(google_question)
# google_question = 'Jak vydělat peníze'
# search(google_question)


# VYGOOGLI Kolik je států v Africe
# VYGOOGLI Najdi mi restauraci v Brně
# VYGOOGLI Jaký je počet obyvatel v Plzni
# VYGOOGLI Kde koupit bitcoin za CZK a jaký je aktuální kurz
# VYGOOGLI Kdo je prezident ve Francii
# VYGOOGLI Kde budou další olympijské hry
# VYGOOGLI Jak vybrat běžky
# VYGOOGLI Jak napsat životopis
# VYGOOGLI Jak rychle zhubnout
# VYGOOGLI Jak vydělat peníze
