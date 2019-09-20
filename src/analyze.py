import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from bert_embedding import BertEmbedding
import re
import numpy as np

all_cards = pd.read_json("data/scryfall-default-cards.json")
all_cards = all_cards[all_cards.lang == "en"]
lsv_scores = pd.read_csv("data/lsv_scores.csv")

cards = lsv_scores.merge(all_cards, on=["set", "name"])[:10]

# Handle `:` -> `.` typo...c'mon LSV
cards["score_clean"] = cards.score.str.replace(
    r"([0-5]):([05])", lambda m: "{}.{}".format(m.group(1), m.group(2))
)
# For now, choose the lower bound of build-around cards as our score
cards["score_clean"] = cards.score_clean.str.replace(
    r"([0-5]\.[05]).*", lambda m: m.group(1)
)
cards["score_clean"] = pd.to_numeric(cards.score_clean, errors="coerce")
cards["build_around"] = cards.score.str.contains("/")

# Split cards are missing `mana_cost`, try getting it from `card_faces`
# cards.loc[cards.mana_cost.isna(), "mana_cost"] = cards[
#     cards.mana_cost.isna()
# ].card_faces.apply(lambda cf: cf[0]["mana_cost"])

# mana_costs_split = cards.mana_cost.apply(lambda mc: re.findall(r"{(.+?)}", mc))


type_line_dummies = cards.type_line.str.get_dummies(" ").add_prefix("type_line_")
rarity_dummies = cards.rarity.str.get_dummies().add_prefix("rarity_")

# TODO: May want to encode these in an ordered or numeric manner as well/instead
power_dummies = cards.power.str.get_dummies().add_prefix("power_")
toughness_dummies = cards.toughness.str.get_dummies().add_prefix("toughness_")

features = pd.concat(
    [
        cards[["score_clean", "cmc"]],
        rarity_dummies,
        type_line_dummies,
        power_dummies,
        toughness_dummies,
    ],
    axis=1,
)
bert = BertEmbedding()
# features["oracle_text"] = cards.oracle_text.fillna("").map(lambda x: bert([x])[0][1])
# features["oracle_text"] = (
#     cards.oracle_text.fillna("")
#     .str.split("\n")
#     .map(lambda x: [embedding for _sentence_items, embedding in bert(x)])
# )
def embed_text(text):
    embedding = bert([text])[0][1]
    if embedding is not None:
        return np.array(pd.Series(embedding).mean())
    return np.zeros(768)


features["oracle_text"] = cards.oracle_text.fillna("").map(embed_text)
features["power"] = pd.to_numeric(cards.power, errors="coerce").fillna(0)
features["toughness"] = pd.to_numeric(cards.toughness, errors="coerce").fillna(0)
features = features.dropna()
train, test = train_test_split(features, test_size=0.2)


# Train on `train`

import pdb

pdb.set_trace()
model = GradientBoostingRegressor().fit(train.drop("score_clean", 1), train.score_clean)

#
# Evaluate `test` into `test.prediction`

test["prediction"] = model.predict(test.drop("score_clean", 1))
test["prediction"] = test.prediction.clip(upper=5.0, lower=0.0)

# Score RMSE

rmse = ((test.prediction - test.score_clean) ** 2).mean() ** 0.5
print("RMSE: {}".format(rmse))
