from typing import Dict


def get_model_path(config: Dict) -> str:
    arr = [
        config["model"]["name"],
        config["dataset"]["name"]
    ]
    if config["model"].get("append_to_name", None) is not None:
        arr.append(config["model"]["append_to_name"])
    return "-".join(arr)

def get_path(config: Dict) -> str:
    return "-".join([
        config["classifier"]["name"],
        str(config["classification"]["num_classes"]),
        str(config["classification"]["num_known_samples_per_class"]),
        str(config["classification"]["num_unknown_samples_per_class"]),
        get_model_path(config)
    ])