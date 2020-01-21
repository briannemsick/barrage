"""
UCI Sentiment Labelled Sentences dataset example.

This examples demonstrates the following:

    1. Writing a custom transformer that passes network params to the network builder.

    2. An NLP dataset.

Notes:
    1. Download: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
    2. Place in a folder named "data/".
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from barrage import BarrageModel
from barrage.api import RecordTransformer
from barrage.utils import io_utils


def get_data():
    """Load sentiment dataset."""
    names = ["sentence", "label"]
    sep = "\t"

    amzn_path = os.path.join("data", "amazon_cells_labelled.txt")
    df_amzn = pd.read_csv(amzn_path, names=names, sep=sep)
    imdb_path = os.path.join("data", "imdb_labelled.txt")
    df_imdb = pd.read_csv(imdb_path, names=names, sep=sep)
    yelp_path = os.path.join("data", "yelp_labelled.txt")
    df_yelp = pd.read_csv(yelp_path, names=names, sep=sep)

    df = pd.concat([df_amzn, df_imdb, df_yelp])
    df_train, records_test = train_test_split(df, test_size=0.2, random_state=42)
    records_train, records_val = train_test_split(
        df_train, test_size=0.2, random_state=42
    )

    return records_train, records_val, records_test


def net(vocab_size, embedding_dim, seq_len, num_classes):
    """Small CNN text classification network."""
    inputs = layers.Input(shape=(seq_len,), name="sequence")
    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=seq_len)(
        inputs
    )
    conv = layers.Conv1D(64, 5, activation="relu")(embedding)
    mp = layers.GlobalMaxPooling1D()(conv)
    dense = layers.Dense(16, activation="relu")(mp)

    # Binary crossentropy vs sparse categorical
    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="target")(dense)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="target")(dense)

    return models.Model(inputs=inputs, outputs=outputs)


class CustomKerasTokenizerWrapper(RecordTransformer):
    """Wrap a Keras preprocessing tokenizer."""

    def __init__(self, mode, loader, input_key, output_key, pad, tokenizer):
        super().__init__(mode, loader)

        self.tokenizer = Tokenizer(**tokenizer)
        self.vocab_size = tokenizer["num_words"]
        self.pad = pad

        self.input_ind = 0
        self.output_ind = 1
        self.input_key = input_key
        self.output_key = output_key

    def fit(self, records):

        # Use the generator API
        # Note because we yield a generator over the loader, the data could have
        # been filepaths
        gen = (
            self.loader.load(records[ii])[self.input_ind][self.input_key][0]
            for ii in range(len(records))
        )
        self.tokenizer.fit_on_texts(gen)

        # Pass the network params to network builder
        self.network_params = {
            "vocab_size": self.vocab_size,
            "seq_len": self.pad["maxlen"],
        }

    def transform(self, data_record):
        # Pad sequences is designed to return a slice of a batch (1, n)
        # We need a (n, )
        text = data_record[self.input_ind][self.input_key]
        transformed_text = self.tokenizer.texts_to_sequences(text)
        padded_text = pad_sequences(transformed_text, **self.pad)

        data_record[self.input_ind][self.input_key] = padded_text[0, :]
        return data_record

    def postprocess(self, score):
        # Threshold 0.5 / Argmax the score

        if len(score) == 1:
            score[self.output_key] = float(score[self.output_key] > 0.5)
        else:
            score[self.output_key] = np.argmax(score[self.output_key])

        return score

    def load(self, path):
        self.tokenizer = io_utils.load_pickle("tokenizer.pkl", path)

    def save(self, path):
        io_utils.save_pickle(self.tokenizer, "tokenizer.pkl", path)


if __name__ == "__main__":
    records_train, records_val, records_test = get_data()

    # Train
    cfg = io_utils.load_json("config_sentiment.json")
    BarrageModel("artifacts").train(cfg, records_train, records_val)

    # Predict
    scores = BarrageModel("artifacts").predict(records_test)
    df_preds = pd.DataFrame(scores)

    acc = (df_preds["target"] == records_test["label"]).mean()
    print(f"Test set accuracy: {acc}")
