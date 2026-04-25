from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_old_model_check_phrase_is_not_used() -> None:
    forbidden = " ".join(["Похоже,", "все", "работает."])
    checked_files = [
        *PROJECT_ROOT.joinpath("docmanage").glob("*.py"),
        *PROJECT_ROOT.joinpath("tests").glob("*.py"),
    ]

    for path in checked_files:
        assert forbidden not in path.read_text(encoding="utf-8")
