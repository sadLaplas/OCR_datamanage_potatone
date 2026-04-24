from pathlib import Path

import torch

from docmanage.ocr_dataset import OcrCharset
from docmanage.ocr_evaluation import (
    DELETE_LABEL,
    INSERT_LABEL,
    build_error_matrix,
    character_error_rate,
    greedy_decode_indexes,
    greedy_decode_logits,
    save_error_matrix_csv,
)


def test_greedy_decode_removes_blank_and_repeats() -> None:
    charset = OcrCharset(("а", "б", "в"))

    decoded = greedy_decode_indexes([0, 1, 1, 0, 2, 2, 3], charset)

    assert decoded == "абв"


def test_greedy_decode_works_with_logits() -> None:
    charset = OcrCharset(("а", "б"))
    logits = torch.tensor(
        [
            [
                [4.0, 1.0, 0.0],
                [0.0, 5.0, 1.0],
                [0.0, 6.0, 1.0],
                [7.0, 0.0, 1.0],
                [0.0, 1.0, 6.0],
            ]
        ]
    )

    decoded = greedy_decode_logits(logits, torch.tensor([5]), charset)

    assert decoded == ["аб"]


def test_character_error_rate_counts_distance() -> None:
    assert character_error_rate("кот", "кот") == 0
    assert character_error_rate("кит", "кот") == 1 / 3


def test_error_matrix_counts_replacement() -> None:
    matrix = build_error_matrix([("кот", "кит")])

    assert matrix["о"]["и"] == 1


def test_error_matrix_counts_deletion() -> None:
    matrix = build_error_matrix([("кот", "ко")])

    assert matrix["т"][DELETE_LABEL] == 1


def test_error_matrix_counts_insertion() -> None:
    matrix = build_error_matrix([("кот", "крот")])

    assert matrix[INSERT_LABEL]["р"] == 1


def test_error_matrix_is_saved_to_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "matrix.csv"
    matrix = build_error_matrix([("кот", "кит")])

    save_error_matrix_csv(matrix, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "target\\prediction" in text
    assert "о" in text
    assert "и" in text
