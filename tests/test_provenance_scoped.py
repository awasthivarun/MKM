"""Unrelated new model definitions do not invalidate model-scoped provenance."""

from mkm.provenance import source_file_hashes


def test_scoped_hash_ignores_new_registry_entries_and_unused_functions(tmp_path):
    source = tmp_path / "mechanisms.py"
    source.write_text('''def shared(x):
    return x + 1

def model_a(x):
    return shared(x)

REGISTRY = {"A": model_a}
''')
    def digest():
        return source_file_hashes(
            tmp_path, ("mechanisms.py",), scoped_symbols={"mechanisms.py": ("REGISTRY", "model_a")},
            scoped_registry_entries={"mechanisms.py": {"REGISTRY": "A"}},
        )
    initial = digest()
    source.write_text(source.read_text() + '''\ndef model_b(x):
    return x ** 2
''')
    source.write_text(source.read_text().replace('"A": model_a}', '"A": model_a, "B": model_b}'))
    assert digest() == initial
    source.write_text(source.read_text().replace('return x + 1', 'return x + 2'))
    assert digest() != initial


def test_unscoped_hash_still_tracks_full_source_file(tmp_path):
    source = tmp_path / "common.py"
    source.write_text("VALUE = 1")
    original = source_file_hashes(tmp_path, ("common.py",))
    source.write_text("VALUE = 2")
    assert source_file_hashes(tmp_path, ("common.py",)) != original
