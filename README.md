# Luis Bot Vargas

## Getting Started

1. Install deps with Pipenv and open a Pipenv shell

```bash
pipenv install
pipenv shell
```

2. Download Scryfall card data

```bash
bin/download_card_data
```

3. Download Channel Fireball articles from the `cfb_articles.csv` manifest

```bash
bin/download_cfb_articles
```

4. Extract LSV rankings from the articles

```bash
bin/extract_cfb_ratings
```

You should now have a bunch of stuff in the `data/` directory


## Progress

### Step 0: Use mean score

As an absolute baseline, we can assume that every card gets the average score
that LSV has handed out, which turns out to be around `2.423502767991948`.


RMSE: 1.142891810708898


### Step 1: Basic regression against attribute subset

Using

- CMC
- All things in the Type Line ("Creature", "Enchantment", "Bear", etc.)
- Rarity


All models run with default options:

Linear Regression:

RMSE: 1.1080096806738646

HistGradientBoostingRegressor:

RMSE: 0.9919545420320893

GradientBoostingRegressor:

RMSE: 0.9604872808415652

RandomForestRegressor:

RMSE: 1.0239968641701904
