import csv

import constants


def percentage(target_count, all_count):
    return f"{target_count/all_count*100:.1f}"


if __name__ == "__main__":
    with open(constants.TRAIN_DATA_PATH) as f:
        # id,keyword,location,text,target
        train_rows = list(csv.DictReader(f))
    with open(constants.TEST_DATA_PATH) as f:
        # id,keyword,location,text
        test_rows = list(csv.DictReader(f))

    # train, testで何件ずつあるか
    train_count, test_count = len(train_rows), len(test_rows)
    all_count = train_count + test_count
    print(
        "Row count: "
        f"train {train_count} ({percentage(train_count, all_count)}%), "
        f"test {test_count} ({percentage(test_count, all_count)}%)"
    )

    # （trainに）正例、負例が何件ずつあるか
    train_pos_count = len([d for d in train_rows if int(d["target"])])
    train_neg_count = len([d for d in train_rows if not int(d["target"])])
    assert train_count == train_pos_count + train_neg_count
    print(
        "In train data: "
        f"positive {train_pos_count} "
        f"({percentage(train_pos_count, train_count)}%), "
        f"negative {train_neg_count} "
        f"({percentage(train_neg_count, train_count)}%)"
    )

    # blankのテキストがあるか
    train_blank_text_count = len([d for d in train_rows if not d["text"]])
    test_blank_text_count = len([d for d in test_rows if not d["text"]])
    print(
        "Blank text: "
        f"in train {train_blank_text_count} "
        f"({percentage(train_blank_text_count, train_count)}%) "
        f"in test {test_blank_text_count} "
        f"({percentage(test_blank_text_count, test_count)}%)"
    )

    # keywordのblank, not blankが何件ずつあるか
    train_blank_keyword_count = len(
        [d for d in train_rows if not d["keyword"]]
    )
    test_blank_keyword_count = len([d for d in test_rows if not d["keyword"]])
    all_blank_keyword_count = (
        train_blank_keyword_count + test_blank_keyword_count
    )
    print(
        f"Blank keyword: all {all_blank_keyword_count} "
        f"({percentage(all_blank_keyword_count, all_count)}%)"
    )
    print(
        f"  in train {train_blank_keyword_count} "
        f"({percentage(train_blank_keyword_count, train_count)}%), "
        f"in test {test_blank_keyword_count} "
        f"({percentage(test_blank_keyword_count, test_count)}%) "
    )

    # locationのblank, not blankが何件ずつあるか
    train_blank_location_count = len(
        [d for d in train_rows if not d["location"]]
    )
    test_blank_location_count = len(
        [d for d in test_rows if not d["location"]]
    )
    all_blank_location_count = (
        train_blank_location_count + test_blank_location_count
    )
    print(
        f"Blank location: all {all_blank_location_count} "
        f"({percentage(all_blank_location_count, all_count)}%)"
    )
    print(
        f"  in train {train_blank_location_count} "
        f"({percentage(train_blank_location_count, train_count)}%), "
        f"in test {test_blank_location_count} "
        f"({percentage(test_blank_location_count, test_count)}%) "
    )

    # textの重複
    train_texts = [d["text"] for d in train_rows]
    test_texts = [d["text"] for d in test_rows]
    all_texts = train_texts + test_texts
    unique_train_texts = set(train_texts)
    unique_test_texts = set(test_texts)
    unique_all_texts = set(all_texts)
    print(
        f"Unique text: all {len(unique_all_texts)} "
        f"({percentage(len(unique_all_texts), all_count)}%)"
    )
    print(
        f"  in train {len(unique_train_texts)} "
        f"({percentage(len(unique_train_texts), train_count)}%) "
        f"in test {len(unique_test_texts)} "
        f"({percentage(len(unique_test_texts), test_count)}%)"
    )
    print(
        f"  same text count in train and test: "
        f"{len(unique_train_texts & unique_test_texts)}"
    )
