import csv
import random
import re
from collections import Counter

import gensim.parsing.preprocessing as gsp
import nltk

import constants

random.seed(20201107)


def normalize_number(text):
    """数字1文字を0に置き換える（桁数は変わらない）"""
    return re.sub(r"\d", "0", text)


def normalize_twitter_shorten_url(text):
    """Twitterの短縮URLを共通の文字列に置き換える（含んでいることを表したい）"""
    return re.sub(r"https?://t\.co/\w+", "SOME_URL", text)


def normalize_twitter_hashtag(text):
    """hashtagを共通の文字列に置き換える（含んでいることを表したい）"""
    return re.sub(r"#\w+", "#hashtag", text)


def normalize_twitter_mention(text):
    """メンションを共通の文字列に置き換える（含んでいることを表したい）"""
    return re.sub(r"@\w+", "@mention", text)


def show_sample(dataset):
    for d in dataset:
        print(d["id"], d["target"], d["text"])
        print("-" * 20)


def show_delimiter():
    print("=" * 40)


def show_more_frequently(counter, topn=20):
    for i, (word, count) in enumerate(counter.most_common(topn)):
        print(f"{word} (count {count})")


preprocess_filters = [
    normalize_number,
    normalize_twitter_shorten_url,
    normalize_twitter_hashtag,
    normalize_twitter_mention,
    gsp.remove_stopwords,
]


if __name__ == "__main__":
    with open(constants.TRAIN_DATA_PATH) as f:
        train_rows = list(csv.DictReader(f))

    sample_rows = random.sample(train_rows, 10)
    print("Check preprocessing")
    show_sample(sample_rows)
    show_delimiter()

    for d in train_rows:
        text = d["text"]
        text = text.lower()
        for filter_func in preprocess_filters:
            text = filter_func(text)
        d["text"] = text

    print("After preprocessing")
    show_sample(sample_rows)

    positive_train_rows = [d for d in train_rows if int(d["target"])]
    print("Positive sample examples")
    show_sample(random.sample(positive_train_rows, 10))
    print("=" * 40)
    print("Negative sample examples")
    negative_train_rows = [d for d in train_rows if not int(d["target"])]
    show_sample(random.sample(negative_train_rows, 10))

    positive_texts = [d["text"] for d in positive_train_rows]
    negative_texts = [d["text"] for d in negative_train_rows]
    print(
        f"Train text count: positive {len(positive_texts)}, "
        f"negative {len(negative_texts)}"
    )
    # 正例／負例それぞれでの頻出語を確認する
    positive_texts_words = [w for d in positive_texts for w in d.split()]
    negative_texts_words = [w for d in negative_texts for w in d.split()]
    wnl = nltk.WordNetLemmatizer()  # 辞書の見出し語化
    positive_texts_words = [wnl.lemmatize(w) for w in positive_texts_words]
    negative_texts_words = [wnl.lemmatize(w) for w in negative_texts_words]
    porter = nltk.PorterStemmer()  # 語幹を取り出す
    positive_texts_words = [porter.stem(w) for w in positive_texts_words]
    print(f"positive texts words {len(positive_texts_words)}")
    negative_texts_words = [porter.stem(w) for w in negative_texts_words]
    print(f"negative texts words {len(negative_texts_words)}")

    positive_word_counter = Counter(positive_texts_words)
    print(f"Frequent words in positive sample (count {len(positive_texts)})")
    show_more_frequently(positive_word_counter, 30)
    show_delimiter()
    print(f"Frequent words in negative sample (count {len(negative_texts)}")
    negative_word_counter = Counter(negative_texts_words)
    show_more_frequently(negative_word_counter, 30)
