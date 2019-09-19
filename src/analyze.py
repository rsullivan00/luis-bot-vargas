import pandas as pd

all_cards = pd.read_json('data/scryfall-default-cards.json')
all_cards = all_cards[all_cards.lang == 'en']
lsv_scores = pd.read_csv('data/lsv_scores.csv')

cards = lsv_scores.merge(all_cards, on=['set', 'name'])
