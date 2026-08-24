from pathlib import Path

import yaml

from mkm.provenance import sha256_file, write_run_metadata


def test_sha256_file_is_content_stable(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_write_run_metadata_round_trips_yaml(tmp_path):
    output = tmp_path / "run_metadata.yaml"
    metadata = {
        "material": "Ag10Pd90",
        "model": "BF_LH",
        "likelihood": "setup_intercept",
        "sampler": {"draws": 1000},
    }

    write_run_metadata(metadata, output)

    with open(output, "r") as file:
        loaded = yaml.safe_load(file)

    assert loaded == metadata
