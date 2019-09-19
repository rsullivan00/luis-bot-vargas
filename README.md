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


Step 1: Linear regression against certain attributes?

TODO: Convert `type_line` into several categories

Things before `-` in type line:
  Land - Y/N
  Instant - Y/N
  Sorcery - Y/N
  Creature - Y/N
  Enchantment - Y/N
  Artifact - Y/N
  Legendary - Y/N

Things after `-` in type line are probably less important, but could be:
  Bird
  Zombie
  Human
  etc.


Instant/Sorcery/Creature/Enchantment/Artifact (or combo of)


Step 2: 
