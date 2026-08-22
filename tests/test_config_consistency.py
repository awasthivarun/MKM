import yaml


def _load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def test_agpd_preprocessing_and_model_metadata_are_consistent():
    preprocessing = _load_yaml("config/preprocessing/agpd_basic.yaml")
    model = _load_yaml("config/models/agpd_basic.yaml")

    assert preprocessing["dataset_name"] == model["dataset_name"]
    assert float(preprocessing["temperature_K"]) == float(model["temperature_K"])
