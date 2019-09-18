from bs4 import BeautifulSoup
import glob
import re
import sys
import pandas as pd
import os


def clean_name(html_name):
    return re.sub(r'[`’]', "'", html_name)

"""
Reads all HTML files in `data/cfb` and extracts card names and their scores to
a CSV.
"""
dfs = []
for filename in glob.glob('data/cfb/*/*.html'):
    print('Processing {}'.format(filename))
    with open(filename) as fp:
        set_name = os.path.basename(os.path.dirname(filename))
        soup = BeautifulSoup(fp, features='html.parser')
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])

        cards = []
        card_name = None
        score = None
        for heading in headings:
            # Scores are prefixed by `Limited: `
            if heading.get_text().startswith('Limited:'):
                score = re.sub(r'Limited:\s*', '', heading.get_text())
                if not len(score):
                    continue
                if card_name is None:
                    print('Warning: Found score without card name. File: {} Score: {}'.format(filename, score), file=sys.stderr)
                    continue
                cards.append([card_name, score])
                card_name = None

            # Card Titles are in Headings and on image alt texts
            next_sibling = heading.find_next_sibling()
            if not next_sibling:
                continue
            card_img = next_sibling.find('img')
            html_name = heading.get_text()
            cleaned_name = clean_name(html_name)
            if not card_img or not card_img['alt'] in (html_name, cleaned_name):
                continue

            card_name = cleaned_name

        df = pd.DataFrame.from_records(cards, columns=['name', 'score'])
        df['set_name'] = set_name
        dfs.append(df)

df = pd.concat(dfs)
df.to_csv('data/lsv_scores.csv')
