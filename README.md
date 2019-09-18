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
