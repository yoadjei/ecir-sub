from src.utils.config import cfg, ROOT


def test_root_points_at_repo():
    assert (ROOT / "config.yaml").exists()


def test_config_has_required_sections():
    c = cfg()
    assert c["seed"] == 13
    assert set(c["systems"]) == {"bm25", "mdpr", "mcontriever", "e5small"}
    for lang, info in c["languages"].items():
        assert info["tier"] in {"low", "mid", "high"}, lang
        assert isinstance(info["dense"], bool), lang


def test_every_e5_language_is_also_a_dense_language():
    c = cfg()
    for lang in c["e5_languages"]:
        assert c["languages"][lang]["dense"] is True


def test_gap_pairs_reference_known_languages():
    c = cfg()
    for hi, lo in c["gap_pairs"]:
        assert hi in c["languages"] and lo in c["languages"]
        assert c["languages"][hi]["tier"] != "low" and c["languages"][lo]["tier"] == "low"
