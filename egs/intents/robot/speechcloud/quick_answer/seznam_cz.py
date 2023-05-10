import re
import requests
from bs4 import BeautifulSoup


def search(question):
    question = question.lower().replace('jaké je', '').replace('jaký je', '').replace('kolik je', '')
    question = re.sub(r'[^\w\s]', '', question).replace(' ', '+')
    url = 'https://search.seznam.cz/?q=' + question
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/64.0.3282.186 Safari/537.36',
        'Accept-Encoding': 'utf-8'}
    r = requests.get(url=url, headers=headers)
    soup = BeautifulSoup(r.content.decode('utf-8'), 'lxml')
    results = [soup.find('ol', class_='e708e4'), soup.find('div', class_='d39249'),
               soup.find('h2', class_='Card-header-title'),
               soup.find('div', class_='c8168c'), soup.find('span', class_='e7cb4d'), soup.find('div', class_='e80211'),
               soup.find('div', class_='bbbafe'), soup.find('p', class_='b73e3a')]
    out = ''
    for result in results:
        if result is not None:
            out = result.text
    return out


