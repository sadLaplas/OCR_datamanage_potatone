from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_state_exists_and_mentions_stage_13() -> None:
    state_path = PROJECT_ROOT / "PROJECT_STATE.md"
    assert state_path.exists()

    text = state_path.read_text(encoding="utf-8")
    assert "Этап 13" in text
    assert "page_text.json" in text
