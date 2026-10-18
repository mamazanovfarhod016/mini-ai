"""Тесты тем самообучения: знание = смысл с формулировками, часть спрятана для экзамена."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import knowledge, store  # noqa: E402
from tests.test_learn import req, srv  # noqa: E402,F401

PARAS = ["главный город франции", "какой город столица франции", "назови столицу франции",
         "что является столицей франции", "столица у франции какая", "в каком городе столица франции",
         "скажи столицу франции", "франция столица", "fransiya poytaxti", "fransiya poytaxti qaysi shahar"]


def test_topic_needs_many_paraphrases(tmp_path):
    with pytest.raises(ValueError):
        knowledge.add_or_merge(tmp_path, "столица франции", PARAS[:3], "Столица Франции — Париж.")


def test_topic_split_heldout_never_trained(tmp_path):
    t, created = knowledge.add_or_merge(tmp_path, "Столица Франции?", PARAS, "Столица Франции — Пари́ж.", "Parij.")
    assert created and t["train"][0] == "столица франции"
    assert len(t["heldout"]) >= 2 and not set(t["heldout"]) & set(t["train"])
    assert t["responses"]["ru"] == ["Столица Франции — Париж."]           # ударение убрано
    intents = knowledge.as_intents(knowledge.load(tmp_path))
    assert not set(t["heldout"]) & set(intents[0]["patterns"])            # экзаменационные — не в обучении


def test_topic_merge_into_existing(tmp_path):
    t, _ = knowledge.add_or_merge(tmp_path, "столица франции", PARAS, "Столица Франции — Париж.")
    more = ["франции столица какая", "где столица франции", "столица франции назови", "столица франции скажи пожалуйста",
            "франция главный город", "столичный город франции", "fransiyaning bosh shahri", "fransiya poytaxti qayer"]
    t2, created = knowledge.add_or_merge(tmp_path, "где столица франции", more, "Столица Франции — Париж.", topic=t["tag"])
    assert not created and len(knowledge.load(tmp_path)) == 1
    assert len(t2["train"]) + len(t2["heldout"]) > 11
    assert t2["responses"]["ru"] == ["Столица Франции — Париж."]


def test_confirm_goes_to_mentor_queue(srv):  # noqa: F811
    base, data = srv
    code, _ = req(base + "/api/confirm", {"q": "столица франции", "a": "Париж — столица Франции.", "web": {"url": "x"}})
    assert code == 200
    item = store.read_json(data / "mentor_inbox.json", [])[0]
    assert item["kind"] == "confirm" and item["status"] == "new"
    assert req(base + "/api/progress")[1] == []
