import csv
import pickle
import profile  # my module

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import constants

SEED = 20201127

if __name__ == "__main__":
    with open(constants.TRAIN_DATA_PATH) as f:
        train_rows = list(csv.DictReader(f))
    print("training dataset (all):", len(train_rows))

    for d in train_rows:
        text = d["text"]
        text = text.lower()
        for filter_func in profile.preprocess_filters:
            text = filter_func(text)
        d["text"] = text

    texts = [d["text"] for d in train_rows]
    targets = [d["target"] for d in train_rows]

    train_texts, val_texts, train_targets, val_targets = train_test_split(
        texts, targets, test_size=0.2, random_state=SEED
    )
    print(f"train: {len(train_targets)}, val: {len(val_targets)}")

    vectorizer = TfidfVectorizer()
    train_X = vectorizer.fit_transform(train_texts)
    val_X = vectorizer.transform(val_texts)
    print(f"train: {train_X.shape}, val: {val_X.shape}")

    with open(constants.TEXT_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(constants.TRAIN_PREPROCESSED_PATH, "wb") as f:
        pickle.dump({"x": train_X, "y": train_targets}, f)
    with open(constants.VAL_PREPROCESSED_PATH, "wb") as f:
        pickle.dump({"x": val_X, "y": val_targets}, f)
