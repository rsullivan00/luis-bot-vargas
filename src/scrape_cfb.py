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
error_count = 0
for filename in glob.glob('data/cfb/*/*.html'):
    print('Processing {}'.format(filename))
    with open(filename) as fp:
        set_name = os.path.basename(os.path.dirname(filename))
        soup = BeautifulSoup(fp, features='html.parser')
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])

        cards = []
        card_name = None
        score = None
        for item in soup.select('a[data-name],h1,h2,h3,h4'):
            # Scores are prefixed by `Limited: `
            if item.get_text().startswith('Limited:'):
                score = re.sub(r'Limited:\s*', '', item.get_text())
                if not len(score):
                    continue
                if card_name is None:
                    error_count += 1
                    print('Warning: Found score without card name. File: {} Score: {}'.format(filename, score), file=sys.stderr)
                    continue
                cards.append([card_name, score])
                card_name = None
                continue

            if not item.get('data-name'):
                continue
            card_name = clean_name(item.get('data-name'))

        df = pd.DataFrame.from_records(cards, columns=['name', 'score'])
        df['set'] = set_name
        dfs.append(df)

df = pd.concat(dfs)
df.to_csv('data/lsv_scores.csv')
if error_count:
    print('{} warnings'.format(error_count))
print('{} card scores found'.format(len(df.index)))
