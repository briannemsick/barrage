import json
import os
import pickle

import numpy as np


def load_json(filename: str, path: str = ""):
    """Load a json object.

    Args:
        filename: str.
        path: str.

    Returns:
        object.
    """
    with open(os.path.join(path, filename), "r") as fn:
        return json.load(fn)


def save_json(obj, filename: str, path: str = ""):
    """Save a json object.

    Args:
        obj: object.
        filename: str.
        path: str.
    """

    def _default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"invalid json serialization type: {type(obj)}")

    with open(os.path.join(path, filename), "w") as fn:
        json.dump(obj, fn, indent=2, sort_keys=True, default=_default)


def load_pickle(filename: str, path: str = ""):
    """Load a pickled object.

    Args:
        filename: str.
        path: str.

    Returns:
        object.
    """
    with open(os.path.join(path, filename), "rb") as fn:
        return pickle.load(fn)


def save_pickle(obj, filename: str, path: str = ""):
    """Save a pickled object.

    Args:
        obj: object.
        filename: str.
        path: str.
    """
    with open(os.path.join(path, filename), "wb") as fn:
        pickle.dump(obj, fn)
