from bs4 import BeautifulSoup
import glob
import re


"""
Reads all HTML files in `data/cfb` and extracts card names and their scores to
a CSV.
"""
for filename in glob.glob('data/cfb/*.html'):
    with open(filename) as fp:
        soup = BeautifulSoup(fp)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])

        cards = []
        card_name = None
        score = None
        for heading in headings:
            # Scores are prefixed by `Limited: `
            if heading.get_text().startswith('Limited:'):
                score = re.sub(r'Limited:\s*', '', heading.get_text())
                if not card_name:
                    raise 'Found score without card name'
                cards.push([card_name, score])

            # Card Titles are in Headings and on image alt texts
            card_img = heading.find_next_sibling().find('img')
            if not card_img or not card_img['alt'] == heading.get_text():
                continue

            card_name = heading.get_text()

        print(cards)


