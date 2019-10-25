import os

import numpy as np
import pandas as pd

from barrage import BarrageModel, api
from barrage.utils import io_utils

NUM_SAMPLES_TRAIN = 407
NUM_SAMPLES_VALIDATION = 193
NUM_SAMPLES_SCORE = 122


def gen_records(num_samples):
    y = np.random.randint(0, 3, num_samples).astype(np.float32)
    x1 = np.random.normal(0, 2.0, num_samples) + y
    x2 = np.random.normal(-1.0, 1.0, num_samples) + y
    x3 = np.random.normal(1.0, 0.5, num_samples) + y
    x4 = np.random.normal(0.5, 0.25, num_samples) + y
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    df["label"] = df["y"].map({0: "class_0", 1: "class_1", 2: "class_2"})
    return df.to_dict(orient="records")


class Transformer(api.RecordTransformer):
    def __init__(self, mode, loader, params):
        super().__init__(mode, loader, params)
        self.output_key = list(self.loader.params["outputs"].values())[0][0]
        self.output_name = list(self.loader.params["outputs"].keys())[0]

    def fit(self, records):
        class_names = list({record[self.output_key] for record in records})
        self.class_map = {ii: class_names[ii] for ii in range(len(class_names))}
        self.inverse_class_map = dict(map(reversed, self.class_map.items()))

        self.network_params = {
            "input_dim": len(list(self.loader.params["inputs"].values())[0]),
            "num_classes": len(class_names),
        }

    def transform(self, record):
        if self.mode == api.RecordMode.TRAIN or self.mode == api.RecordMode.VALIDATION:
            val = self.inverse_class_map[record[1][self.output_name][0]]
            record[1][self.output_name] = np.array(val)

        return record

    def postprocess(self, score):
        ind = np.argmax(score[self.output_name])
        score[self.output_name] = self.class_map[ind]
        return score

    def save(self, path):
        io_utils.save_pickle(self.class_map, "class_map.pkl", path)
        io_utils.save_pickle(self.inverse_class_map, "inverse_class_map.pkl", path)

    def load(self, path):
        self.class_map = io_utils.load_pickle("class_map.pkl", path)
        self.inverse_class_map = io_utils.load_pickle("inverse_class_map.pkl", path)


def test_simple_output(artifact_dir):
    records_train = gen_records(NUM_SAMPLES_TRAIN)
    records_validation = gen_records(NUM_SAMPLES_VALIDATION)
    records_score = gen_records(NUM_SAMPLES_SCORE)

    loc = os.path.abspath(os.path.dirname(__file__))
    cfg = io_utils.load_json("config_single_output.json", loc)

    bm = BarrageModel(artifact_dir)
    bm.train(cfg, records_train, records_validation)
    scores = bm.predict(records_score)

    df_scores = pd.DataFrame(scores)
    records_score = pd.DataFrame(records_score)
    assert (df_scores["softmax"] == records_score["label"]).mean() >= 0.90
