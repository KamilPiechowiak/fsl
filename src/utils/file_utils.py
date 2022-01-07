from typing import Any, Dict
import json
import pickle
import os

def save_json(d: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(d, file)

def read_json(path: str) -> Dict:
    with open(path) as file:
        return json.load(file)

def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(obj, file)

def read_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)