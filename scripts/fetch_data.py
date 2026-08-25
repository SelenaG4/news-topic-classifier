"""Fetches the AG News topic-classification dataset and writes it to
data/*.csv in the same format already checked into this repo.

AG News is a well-known public benchmark (Zhang, Zhao & LeCun, 2015):
120,000 training articles and 7,600 test articles, evenly split across four
topics -- World, Sports, Business, Sci/Tech -- each row a (class, title,
description) triple built from newswire headlines/snippets.

Normally the easiest way to pull this is Hugging Face's `datasets` library
(`load_dataset("ag_news")`), which is exactly what the Colab notebook in
notebooks/ uses. This script instead pulls the same canonical data from a
plain CSV mirror on GitHub, because Hugging Face's own domain
(huggingface.co) is not reachable from the sandbox this project was
originally built in -- documented in the README's "The Hugging Face Hub
constraint -- and how it's handled" section. If you're running this from a
normal machine with unrestricted internet, either source gives you the
identical dataset; this script just doesn't require the
`datasets`/`huggingface_hub` packages.

Usage:
    python scripts/fetch_data.py
"""
from __future__ import annotations

import csv
import random
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
TEST_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

PER_CLASS_TRAIN_SAMPLE = 10000  # 4 classes x 10,000 = 40,000-row training sample checked into data/
RANDOM_SEED = 42


def _download_csv_rows(url: str) -> list[list[str]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return list(csv.reader(text.splitlines()))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Downloading training data from {TRAIN_URL} ...")
    train_rows = _download_csv_rows(TRAIN_URL)
    print(f"  {len(train_rows)} rows")

    print(f"Downloading test data from {TEST_URL} ...")
    test_rows = _download_csv_rows(TEST_URL)
    print(f"  {len(test_rows)} rows")

    # Stratified sample of the training set -- full 120k rows is real and
    # usable, but checking a ~30MB CSV into a portfolio repo isn't worth it
    # when a balanced 40k-row sample trains the classical baseline just as
    # well in a fraction of the time. The full training set is one re-run
    # of this script away, and the Colab notebook fine-tunes the transformer
    # on the complete 120k rows via `datasets`, not this sample.
    random.seed(RANDOM_SEED)
    by_class: dict[str, list[list[str]]] = {}
    for row in train_rows:
        by_class.setdefault(row[0], []).append(row)

    sample = []
    for cls, rows in sorted(by_class.items()):
        random.shuffle(rows)
        sample.extend(rows[:PER_CLASS_TRAIN_SAMPLE])
    random.shuffle(sample)

    train_out = DATA_DIR / "ag_news_train_sample.csv"
    with train_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "title", "description"])
        writer.writerows(sample)
    print(f"Wrote {len(sample)} rows to {train_out}")

    test_out = DATA_DIR / "ag_news_test.csv"
    with test_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "title", "description"])
        writer.writerows(test_rows)
    print(f"Wrote {len(test_rows)} rows to {test_out}")


if __name__ == "__main__":
    main()
