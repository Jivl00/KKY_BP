import re
import requests
from bs4 import BeautifulSoup

question = 'Kdy se narodil Trump?'
# question = 'Datum narozeni kralovny Alzbety'
# question = 'hlavní město makedonie'
# question = 'hlavní město cech'
# question = 'jake je hlavní město cech'
question = 'pocasi plzen'
# question = 'pocasi slovensko'
# question = 'jake je pocasi slovensko'
# question = 'jaky je kurz eura'
# question = 'kolik je 100 kč na eura'
# question = 'jaky je aktualni cas'
# question = 'Co to je recese'
# question = 'pes anglicky'

print(question)
question = question.lower().replace('jake je', '').replace('jaky je', '').replace('kolik je', '')
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
if out == '':
    print('Neznám odpověď')
else:
    print(out)
