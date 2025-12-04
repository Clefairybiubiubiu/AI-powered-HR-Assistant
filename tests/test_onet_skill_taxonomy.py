from pathlib import Path

from resume_matcher.skill_taxonomy_onet import ONETSkillTaxonomy


def test_onet_taxonomy_instantiates(tmp_path):
    taxonomy = ONETSkillTaxonomy(data_dir=tmp_path, enable_cache=False)
    assert taxonomy.data_dir == tmp_path


def test_cache_directory_exists():
    cache_dir = Path("cache") / "onet"
    assert cache_dir.exists(), "cache/onet directory should exist"


def test_methods_exist_and_return_defaults():
    taxonomy = ONETSkillTaxonomy(enable_cache=False)
    assert isinstance(taxonomy.get_related_skills("SQL"), list)
    assert taxonomy.get_skill_category("SQL") is None
    expanded = taxonomy.expand_skill("Python")
    assert isinstance(expanded, dict)
    assert "skill" in expanded and expanded["skill"] == "Python"

