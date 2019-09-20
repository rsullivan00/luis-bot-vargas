import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

all_cards = pd.read_json("data/scryfall-default-cards.json")
all_cards = all_cards[all_cards.lang == "en"]
lsv_scores = pd.read_csv("data/lsv_scores.csv")

cards = lsv_scores.merge(all_cards, on=["set", "name"])

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
# 444 dummies total
type_line_dummies = cards.type_line.str.get_dummies(" ").add_prefix("type_line_")
rarity_dummies = cards.rarity.str.get_dummies().add_prefix("rarity_")

features = pd.concat(
    [cards[["score_clean", "cmc"]], rarity_dummies, type_line_dummies], axis=1
)
features = features.dropna()
train, test = train_test_split(features, test_size=0.2)


# Train on `train`


model = RandomForestRegressor().fit(train.drop("score_clean", 1), train.score_clean)


# Evaluate `test` into `test.prediction`

test["prediction"] = model.predict(test.drop("score_clean", 1))
test["prediction"] = test.prediction.clip(upper=5.0, lower=0.0)

# Score RMSE

rmse = ((test.prediction - test.score_clean) ** 2).mean() ** 0.5
print("RMSE {}".format(rmse))
