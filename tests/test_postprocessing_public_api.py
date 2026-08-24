import mkm.postprocessing as postprocessing


def test_postprocessing_all_exports_exist():
    missing = [name for name in postprocessing.__all__ if not hasattr(postprocessing, name)]
    assert missing == []


def test_postprocessing_all_has_no_duplicates():
    assert len(postprocessing.__all__) == len(set(postprocessing.__all__))
