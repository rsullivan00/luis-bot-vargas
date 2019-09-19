import pandas as pd

all_cards = pd.read_json('data/scryfall-default-cards.json')
all_cards = all_cards[all_cards.lang == 'en']
lsv_scores = pd.read_csv('data/lsv_scores.csv')

cards = lsv_scores.merge(all_cards, on=['set', 'name'])

# For now, choose the lower bound of build-around cards as our score
cards['score_clean'] = cards.score.str.replace(r'([0-5]\.[05]).*', lambda m: m.group(1))
cards['score_clean'] = pd.to_numeric(cards.score_clean)

import pdb; pdb.set_trace()
