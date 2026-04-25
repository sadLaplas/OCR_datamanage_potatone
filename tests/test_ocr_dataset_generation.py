import json
from pathlib import Path

import pytest

from docmanage.config import load_config, prepare_directories
from docmanage.ocr_dataset_generation import OcrDatasetError, generate_ocr_dataset


def test_generate_dataset_demo_mode_creates_files(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    output_dir = tmp_path / "ocr_demo"

    result = generate_ocr_dataset(config, output_dir=output_dir, demo=True, seed=7)

    assert result.total_samples == 12
    assert result.train_samples + result.val_samples == 12
    assert result.train_annotations_path.exists()
    assert result.val_annotations_path.exists()
    assert result.metadata_path.exists()

    train_records = read_jsonl(result.train_annotations_path)
    val_records = read_jsonl(result.val_annotations_path)

    assert train_records
    assert val_records
    assert (output_dir / train_records[0]["image_path"]).exists()
    assert (output_dir / val_records[0]["image_path"]).exists()
    assert train_records[0]["text"]
    assert train_records[0]["split"] == "train"
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["generation_mode"] == "clean"


def test_generate_dataset_keeps_image_and_text_pair(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    output_dir = tmp_path / "ocr_pairs"

    result = generate_ocr_dataset(
        config,
        output_dir=output_dir,
        count=8,
        val_ratio=0.25,
        seed=11,
    )
    records = read_jsonl(result.train_annotations_path) + read_jsonl(result.val_annotations_path)

    assert len(records) == 8
    for record in records:
        assert record["sample_id"].startswith("sample_")
        assert record["text_length"] == len(record["text"])
        assert (output_dir / record["image_path"]).exists()


def test_generate_dataset_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_result = generate_ocr_dataset(
        config,
        output_dir=first_dir,
        count=10,
        val_ratio=0.2,
        seed=99,
    )
    second_result = generate_ocr_dataset(
        config,
        output_dir=second_dir,
        count=10,
        val_ratio=0.2,
        seed=99,
    )

    assert first_result.fonts_used == second_result.fonts_used
    assert first_result.train_samples == second_result.train_samples
    assert first_result.val_samples == second_result.val_samples
    assert first_result.train_annotations_path.read_text(encoding="utf-8") == (
        second_result.train_annotations_path.read_text(encoding="utf-8")
    )
    assert first_result.val_annotations_path.read_text(encoding="utf-8") == (
        second_result.val_annotations_path.read_text(encoding="utf-8")
    )


def test_generate_dataset_raises_for_bad_output_path(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    output_file = tmp_path / "dataset.json"
    output_file.write_text("busy", encoding="utf-8")

    with pytest.raises(OcrDatasetError, match="занята файлом"):
        generate_ocr_dataset(config, output_dir=output_file, demo=True)


def test_generate_dataset_raises_for_bad_count(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)

    with pytest.raises(OcrDatasetError, match="хотя бы 2 примера"):
        generate_ocr_dataset(config, output_dir=tmp_path / "ocr_bad", count=1)


def test_generate_dataset_realistic_mode_creates_metadata(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)
    output_dir = tmp_path / "ocr_realistic"

    result = generate_ocr_dataset(
        config,
        output_dir=output_dir,
        demo=True,
        seed=19,
        mode="realistic",
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    records = read_jsonl(result.train_annotations_path) + read_jsonl(result.val_annotations_path)

    assert result.generation_mode == "realistic"
    assert metadata["generation_mode"] == "realistic"
    assert metadata["realistic_effects"]
    assert records
    assert (output_dir / records[0]["image_path"]).exists()


def test_generate_dataset_raises_for_unknown_mode(tmp_path: Path) -> None:
    config = prepare_test_config(tmp_path)

    with pytest.raises(OcrDatasetError, match="Режим генерации"):
        generate_ocr_dataset(
            config,
            output_dir=tmp_path / "ocr_unknown",
            demo=True,
            mode="too_much",
        )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_test_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: docmanage-test",
                "run_mode: test",
                "data_dir: data",
                "artifacts_dir: artifacts",
                "temp_dir: tmp",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    prepare_directories(config)
    return config
