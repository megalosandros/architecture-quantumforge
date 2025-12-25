import os
import re
import requests
import time
from bs4 import BeautifulSoup

BASE_URL = "https://gameofthrones.fandom.com/ru/wiki/" # Base fandom wiki URL
TARGET_DIR = "original" # Download directory
MIN_TEXT_LENGTH = 40 # To exclude redundant headers
MAX_TEXT_LENGTH = 1000 # To exclude large technical paragraphs or auto generated data
SECONDS_SLEEP = 1


# Create original data directory
os.makedirs(TARGET_DIR, exist_ok=True)


# Fandom wiki page names
pages = [
    "Джон_Сноу",
    "Дейнерис_Таргариен",
    "Арья_Старк",
    "Серсея_Ланнистер",
    "Бран_Старк",
    "Джейме_Ланнистер",
    "Санса_Старк",
    "Теон_Грейджой",
    "Тирион_Ланнистер",
    "Джорах_Мормонт",
    "Джоффри_Баратеон",
    "Мелисандра",
    "Кейтилин_Старк",
    "Маргери_Тирелл",
    "Петир_Бейлиш",
    "Рамси_Болтон",
    "Робб_Старк",
    "Роберт_Баратеон",
    "Сандор_Клиган",
    "Эддард_Старк",
    "Винтерфелл",
    "Королевская_Гавань",
    "Чёрный_замок",
    "Драконий_Камень",
    "Старомест",
    "Орлиное_Гнездо",
    "Пайк",
    "Харренхол",
    "Дредфорт",
    "Риверран",
    "Хайгарден",
    "Утёс_Кастерли",
    "Браавос",
    "Ваес_Дотрак",
    "Волантис",
    "Астапор"
]


# Remove redundant spaces, links and citations
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove redundant spaces
    text = re.sub(r'\[\d+\]', '', text) # remove links
    text = re.sub(r'\[источник\?\]', '', text, flags=re.IGNORECASE)  # ← русская «[источник?]»
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE) # remove citations
    text = re.sub(r'(Править|Edit)\s+', ' ', text)  # remove specific Wiki marks
    return text.strip()


# Extract only article text
def extract_main_content(soup):
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return ""

    # Remove all redundant content
    for elem in content_div.find_all(['table', 'div', 'aside', 'nav', 'ul', 'dl']):
        if len(elem.contents) == 0:
            continue

        classes = elem.get('class', [])
        if not classes:
            continue

        if any(c in classes for c in [
            'toc', 'sidebar', 'infobox', 'navbox', 'gallery',
            'mw-references-wrap', 'printfooter', 'catlinks', 'hatnote'
        ]):
            elem.decompose()

    # Remove all links
    for a in content_div.find_all('a'):
        a.unwrap()

    # Prepare final text
    paragraphs = []
    for p in content_div.find_all('p'):
        txt = clean_text(p.get_text())
        if MIN_TEXT_LENGTH <= len(txt) <= MAX_TEXT_LENGTH:
            paragraphs.append(txt)

    return '\n\n'.join(paragraphs)


# Download pages:
for page in pages:
    url = BASE_URL + page
    filename = re.sub(r'[<>:"/\\|?*]', '_', page) + ".txt"  # безопасное имя файла
    filepath = os.path.join("", "original", filename)

    print(f"Downloading {page} to {filename}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # Page title
        title_tag = soup.find('h1')
        title = clean_text(title_tag.get_text()) if title_tag else page.replace('_', ' ')

        # Main text
        main_text = extract_main_content(soup)
        if not main_text:
            main_text = clean_text(soup.get_text())

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Source: {url}\n# Title: {title}\n\n{main_text}")

        print(f"OK: {len(main_text)} characters saved")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(SECONDS_SLEEP) # to not torture wiki

print("\ndone\n")