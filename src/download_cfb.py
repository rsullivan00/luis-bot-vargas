import pandas as pd
import os
import re
import requests

ARTICLE_PREFIX='https://www.channelfireball.com/articles'
CFB_DATA_LOCATION='data/cfb'

data = pd.read_csv('cfb_articles.csv')
for row in data.itertuples():
    url = row.url
    setname = row.set
    article_name = url.strip('/').split('/')[-1]
    set_dir = '{}/{}'.format(CFB_DATA_LOCATION, setname)
    os.makedirs(set_dir, exist_ok=True)
    html_name = '{}/{}.html'.format(set_dir, article_name)
    if os.path.isfile(html_name):
        continue

    print('Downloading {}'.format(url))
    with open(html_name, 'w') as html_file:
        html_file.write(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text)
