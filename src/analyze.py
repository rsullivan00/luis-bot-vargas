import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import re
import numpy as np

all_cards = pd.read_json("data/scryfall-default-cards.json")
all_cards = all_cards[all_cards.lang == "en"]
lsv_scores = pd.read_csv("data/lsv_scores.csv")

cards = lsv_scores.merge(all_cards, on=["set", "name"])

# Handle `:` -> `.` typo...c'mon LSV
cards["score_clean"] = cards.score.str.replace(
    r"([0-5]):([05])", lambda m: "{}.{}".format(m.group(1), m.group(2)))
# For now, choose the lower bound of build-around cards as our score
cards["score_clean"] = cards.score_clean.str.replace(r"([0-5]\.[05]).*",
                                                     lambda m: m.group(1))
cards["score_clean"] = pd.to_numeric(cards.score_clean, errors="coerce")
cards["build_around"] = cards.score.str.contains("/")

features = pd.DataFrame()
features["power"] = pd.to_numeric(cards.power, errors="coerce").fillna(0)
features["toughness"] = pd.to_numeric(cards.toughness,
                                      errors="coerce").fillna(0)


def mana_tokenizer(s):
    tokens = []
    for mana_item in re.findall(r"{(.+?)}", s):
        if mana_item.isdigit():
            tokens.extend(["colorless"] * int(mana_item))
        else:
            tokens.append(mana_item)
    return tokens


# Split cards are missing `mana_cost`, try getting it from `card_faces`
cards.loc[cards.mana_cost.isna(), "mana_cost"] = cards[
    cards.mana_cost.isna()].card_faces.apply(lambda cf: cf[0]["mana_cost"])

mana_vectorizer = CountVectorizer(tokenizer=mana_tokenizer)
vectorized_mana = mana_vectorizer.fit_transform(cards.mana_cost)
mana_cost_df = pd.DataFrame(
    vectorized_mana.todense(),
    columns=mana_vectorizer.get_feature_names()).add_prefix("mana_cost_")
features = features.join(mana_cost_df)

ngram_counter = CountVectorizer(ngram_range=(1, 1))
oracle_ngrams = ngram_counter.fit_transform(cards.oracle_text.fillna(""))
oracle_ngrams_df = pd.DataFrame(oracle_ngrams.todense()).add_prefix("ngram_")
features = features.join(oracle_ngrams_df)

type_line_dummies = cards.type_line.str.get_dummies(" ").add_prefix(
    "type_line_")
rarity_dummies = cards.rarity.str.get_dummies().add_prefix("rarity_")

# TODO: May want to encode these in an ordered or numeric manner as well/instead
power_dummies = cards.power.str.get_dummies().add_prefix("power_")
toughness_dummies = cards.toughness.str.get_dummies().add_prefix("toughness_")

features = pd.concat(
    [
        features, rarity_dummies, type_line_dummies, power_dummies,
        toughness_dummies
    ],
    axis=1,
)

features = features[cards.score_clean.notna()]
train, test = train_test_split(features, test_size=0.2)
train = train.copy()
test = test.copy()

# Params to search
#  GBR depth - (3 - 8 maybe)
#  GBR min samp leaf
#  GBR min samp split
#  n estimators
#  Ngram range

# Train on `train`

n_estimators = [10, 20, 35, 50, 75, 100, 150, 200]

test_results = []
train_results = []

for estimator in n_estimators:
    model = GradientBoostingRegressor(n_estimators=estimator).fit(
        train, cards.score_clean.loc[train.index])

    train_pred = model.predict(train).clip(0, 5)
    train_rmse = ((train_pred -
                   cards.score_clean.loc[train.index])**2).mean()**0.5
    train_results.append(train_rmse)

    test_pred = model.predict(test).clip(0, 5)
    test_rmse = ((test_pred -
                  cards.score_clean.loc[test.index])**2).mean()**0.5
    test_results.append(test_rmse)

line1, = plt.plot(n_estimators, train_results, 'b', label='Train RMSE')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE')
plt.xlabel('n_estimators')
plt.show()
