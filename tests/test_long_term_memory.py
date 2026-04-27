import json

from loop_core import LongTermMemory, LongTermMemoryConfig


def test_storage_path_accepts_string(tmp_path):
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=str(tmp_path / "k.jsonl")))
    ltm.consolidate([{"text": "hello"}])
    assert (tmp_path / "k.jsonl").exists()


def test_consolidate_appends_jsonl(tmp_path):
    path = tmp_path / "k.jsonl"
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=path))
    ltm.consolidate([{"text": "a"}, {"text": "b"}])
    ltm.consolidate([{"text": "c"}])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["a", "b", "c"]


def test_load_recent_returns_tail(tmp_path):
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=tmp_path / "k.jsonl"))
    for i in range(5):
        ltm.consolidate([{"text": f"item-{i}"}])
    recent = ltm.load_recent(limit=2)
    assert [r["text"] for r in recent] == ["item-3", "item-4"]


def test_truncate_enforces_max_entries(tmp_path):
    ltm = LongTermMemory(
        LongTermMemoryConfig(storage_path=tmp_path / "k.jsonl", max_entries=3)
    )
    for i in range(6):
        ltm.consolidate([{"text": f"item-{i}"}])

    lines = (tmp_path / "k.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines]
    assert texts == ["item-3", "item-4", "item-5"]


def test_load_recent_empty_when_file_missing(tmp_path):
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=tmp_path / "missing.jsonl"))
    # The dir is created but nothing has been written yet.
    (tmp_path / "missing.jsonl").unlink(missing_ok=True)
    assert ltm.load_recent() == []


def test_records_preserve_unicode_separators(tmp_path):
    """Regression: str.splitlines() splits on U+2028/U+2029, str.split('\\n') doesn't."""
    path = tmp_path / "k.jsonl"
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=path))
    payload = "first line second line"  # contains U+2028
    ltm.consolidate([{"text": payload}])
    ltm.consolidate([{"text": "next"}])

    recent = ltm.load_recent()
    assert len(recent) == 2
    assert recent[0]["text"] == payload


def test_existing_file_line_count_preserved(tmp_path):
    path = tmp_path / "k.jsonl"
    path.write_text("\n".join(json.dumps({"text": f"x{i}"}) for i in range(4)) + "\n")
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=path, max_entries=5))
    ltm.consolidate([{"text": "y"}])
    ltm.consolidate([{"text": "z"}])  # should now trigger truncation
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert json.loads(lines[-1])["text"] == "z"
