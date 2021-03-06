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


### Step 2: Add Power/Toughness


#### Using one-hot encoding for both

GradientBoostingRegressor:

RMSE: 0.9534546149704877

#### Using numeric (coercing to 0)

GradientBoostingRegressor:

RMSE: 0.9475962823081454

#### Using numeric (coercing `*` to 0) + one-hot

GradientBoostingRegressor:

RMSE: 0.922496900426324

### Step 3: Add Mana Cost

#### One-hot encoded

GradientBoostingRegressor:

RMSE: 0.9838816376488838

### Step 4: Add BERT oracle text encoding

RMSE: 0.902928983016964


### Digging into errors

Model is more wrong on build around cards (makes sense because we're ditching
the high-end score)

```
(Pdb) test.groupby('build_around').abs_errors.mean()
build_around
False    0.734701
True     1.191530
Name: abs_errors, dtype: float64
```

Performs worst on War of the Spark, one of the more complex sets

```
(Pdb) test.groupby('set').abs_errors.mean()
set
aer    0.689364
akh    0.794437
dom    0.829246
grn    0.633549
hou    0.859139
kld    0.715277
m20    0.836698
rix    0.565465
rna    0.789332
war    0.970275
xln    0.772494
Name: abs_errors, dtype: float64
```

Errors are larger on bad and good cards

```
(Pdb) test.groupby('score_clean').abs_errors.mean()
score_clean
0.0    1.886978
0.5    1.243028
1.0    1.035024
1.5    0.703808
2.0    0.411065
2.5    0.365892
3.0    0.609503
3.5    0.774214
4.0    1.053000
4.5    1.338389
5.0    1.732376
Name: abs_errors, dtype: float64
```

Probably because it's not giving out 0s or 5s

```
(Pdb) test.prediction.describe()
count    497.000000
mean       2.389886
std        0.583950
min        0.713418
25%        1.964401
50%        2.334365
75%        2.753770
max        4.165945
```

Which it should be giving to rares and mythics, but isn't.

```
(Pdb) test.groupby('rarity').abs_errors.mean()
rarity
common      0.649518
mythic      1.029016
rare        0.937026
uncommon    0.780998
Name: abs_errors, dtype: float64
(Pdb) test.groupby('rarity').prediction.describe()
          count      mean       std       min       25%       50%       75%       max
rarity
common    208.0  2.101130  0.409568  0.713418  1.845938  2.131054  2.373937  3.136085
mythic     27.0  3.162709  0.681181  1.543607  2.619579  3.415914  3.542432  4.165945
rare      102.0  2.592148  0.661875  1.276773  2.084486  2.587752  3.197760  3.658593
uncommon  160.0  2.505911  0.490831  1.517939  2.159083  2.525716  2.886970  3.523332
```

It really sucks at evaluating planeswalkers

```
(Pdb) test.groupby('type_line_Planeswalker').abs_errors.mean()
type_line_Planeswalker
0    0.758768
1    1.389938
Name: abs_errors, dtype: float64
```

But is better at Creatures

```
(Pdb) test.groupby('type_line_Creature').abs_errors.mean()
type_line_Creature
0    0.917510
1    0.641529
Name: abs_errors, dtype: float64
```

### Step 5: Improve mana encoding

#### Encoding color-specific

Mana is now encoded as a series of integer features like:

```
mana_cost_colorless: 2
mana_cost_b: 2 # Black
mana_cost_w: 2 # White
...
mana_cost_b/w: 0 # B/W hybrid mana
...
```

RMSE: 0.9269636374887835

### Step 6: Try n-gram encoding

2-4 grams

RMSE: 0.9193461511900894

1-4 grams

RMSE: 0.9527470749862402


#### Next

- Grid search in sklearn to improve hyperparams
- Test out if bag of words performs as well/better than BERT
    - Do it without ngrams
    - Add ngrams on top


#### Grammar

Thoughts on the oracle text structure:

*Keyword abilities*

Sometimes, keyword abilities are separated by newlines
```
Flash
Flying
Dreamcaller Siren can block only creatures with flying.
When Dreamcaller Siren enters the battlefield, if you control another Pirate, tap up to two target nonland permanents.
```

Sometimes, keyword abilities are templated together, comma delimited
```
Flying, menace
```
```
This spell can't be countered.
Trample, hexproof
```


*Activated Abilities*

They always have <cost>: <effect>
```
{X}{G}{G}: Put X +1/+1 counters on target land you control. That land becomes a 0/0 Elemental creature with haste. It's still a land.
```

Multiple costs are comma-separated
```
{T}, Discard a card: Draw a card.
```
```
{7}{U}, {T}, Sacrifice Shore Keeper: Draw three cards.
```

Multiple activated abilties are newline-separated
```
{T}: Add {C}.
{1}, {T}: Add one mana of any color.
```


*Triggered abilities*

```
Whenever an opponent discards a card, that player loses 2 life.
Raid — At the beginning of your end step, if you attacked with a creature this turn, target opponent discards a card.
```

```
When Jungleborn Pioneer enters the battlefield, create a 1/1 blue Merfolk creature token with hexproof. (It can't be the target of spells or abilities your opponents control.)
```

Typically begin with clauses like `whenever x, y happens`.

- `When ~ enters the battlefield`
- `When ~ dies`
- `Whenever you draw a card`
- `Whenever you discard a card`

*Planeswalkers*

Follow the `<cost>: <effect>` format, where costs refer to loyalty changes.

```
+2: Create a 2/2 black Pirate creature token with menace.
−3: Destroy target artifact, creature, or enchantment. Create a Treasure token. (It's an artifact with "{T}, Sacrifice this artifact: Add one mana of any color.")
−10: Target player's life total becomes 1.
```

With static abilies

```
Activated abilities of artifacts your opponents control can't be activated.
+1: Until your next turn, up to one target noncreature artifact becomes an artifact creature with power and toughness each equal to its converted mana cost.
−2: You may choose an artifact card you own from outside the game or in exile, reveal that card, and put it into your hand.
```

*"Choose" effects*


```
Choose one or both —
• Target creature gets -1/-1 until end of turn.
• Put a +1/+1 counter on target creature.
```
