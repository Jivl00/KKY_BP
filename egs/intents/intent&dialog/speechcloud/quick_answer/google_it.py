"""
NOTE: Google constantly keeps on changing its SERP structure and overall algorithm
so this code will eventually stop working
"""

import re
import requests
from bs4 import BeautifulSoup


def search(google_question):
    if 'počasí' in google_question:
        from speechcloud.quick_answer import seznam_cz
        return seznam_cz.search(google_question)
    google_question = re.sub(r'[^\w\s]', '', (google_question.lower())).replace(' ', '%20')
    url = 'https://www.google.com/search?q=' + google_question
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
            return result.text
    return ''

