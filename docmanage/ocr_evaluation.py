from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .ocr_dataset import OcrCharset

INSERT_LABEL = "<ins>"
DELETE_LABEL = "<del>"


class OcrEvaluationError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class OcrPredictionExample:
    sample_id: str
    target: str
    prediction: str
    cer: float


def greedy_decode_logits(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    charset: OcrCharset,
) -> list[str]:
    if logits.ndim != 3:
        raise OcrEvaluationError("Выход модели должен быть трехмерным.")
    if output_lengths.ndim != 1 or output_lengths.shape[0] != logits.shape[0]:
        raise OcrEvaluationError("Длины выхода не подходят к батчу.")

    best_indexes = logits.argmax(dim=2).detach().cpu()
    decoded_texts: list[str] = []

    for row_index, row in enumerate(best_indexes):
        length = int(output_lengths[row_index].item())
        decoded_texts.append(greedy_decode_indexes(row[:length].tolist(), charset))

    return decoded_texts


def greedy_decode_indexes(indexes: list[int], charset: OcrCharset) -> str:
    chars: list[str] = []
    previous_index: int | None = None

    # CTC сначала схлопывает повторы, а потом убирает пустой класс.
    for index in indexes:
        if index == previous_index:
            continue
        previous_index = index

        if index == charset.blank_index:
            continue
        if 1 <= index <= len(charset.characters):
            chars.append(charset.characters[index - 1])

    return "".join(chars)


def character_error_rate(prediction: str, target: str) -> float:
    if not target:
        return 0.0 if not prediction else float(len(prediction))

    return levenshtein_distance(target, prediction) / len(target)


def levenshtein_distance(source: str, target: str) -> int:
    previous_row = list(range(len(target) + 1))

    for source_index, source_char in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_char in enumerate(target, start=1):
            replace_cost = 0 if source_char == target_char else 1
            current_row.append(
                min(
                    previous_row[target_index] + 1,
                    current_row[target_index - 1] + 1,
                    previous_row[target_index - 1] + replace_cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]


def align_strings(target: str, prediction: str) -> list[tuple[str | None, str | None]]:
    rows = len(target) + 1
    columns = len(prediction) + 1
    distances = [[0] * columns for _ in range(rows)]

    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            replace_cost = 0 if target[row - 1] == prediction[column - 1] else 1
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + replace_cost,
            )

    aligned: list[tuple[str | None, str | None]] = []
    row = len(target)
    column = len(prediction)

    # Идем назад по таблице Левенштейна и получаем замены, пропуски и вставки.
    while row > 0 or column > 0:
        if row > 0 and column > 0:
            replace_cost = 0 if target[row - 1] == prediction[column - 1] else 1
            if distances[row][column] == distances[row - 1][column - 1] + replace_cost:
                aligned.append((target[row - 1], prediction[column - 1]))
                row -= 1
                column -= 1
                continue

        if row > 0 and distances[row][column] == distances[row - 1][column] + 1:
            aligned.append((target[row - 1], None))
            row -= 1
            continue

        aligned.append((None, prediction[column - 1]))
        column -= 1

    aligned.reverse()
    return aligned


def build_error_matrix(
    pairs: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}

    for target, prediction in pairs:
        for target_char, predicted_char in align_strings(target, prediction):
            row_label = target_char if target_char is not None else INSERT_LABEL
            column_label = predicted_char if predicted_char is not None else DELETE_LABEL
            if row_label == column_label:
                continue
            matrix.setdefault(row_label, {})
            matrix[row_label][column_label] = matrix[row_label].get(column_label, 0) + 1

    return matrix


def save_error_matrix_csv(
    matrix: dict[str, dict[str, int]],
    path: Path,
) -> None:
    labels = collect_matrix_labels(matrix)

    try:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["target\\prediction", *labels])
            for row_label in labels:
                row = [row_label]
                for column_label in labels:
                    row.append(matrix.get(row_label, {}).get(column_label, 0))
                writer.writerow(row)
    except OSError as error:
        raise OcrEvaluationError("Не получилось сохранить матрицу ошибок.") from error


def save_prediction_examples(
    examples: list[OcrPredictionExample],
    path: Path,
    epoch: int,
) -> None:
    try:
        with path.open("w", encoding="utf-8") as file:
            for example in examples:
                payload = {
                    "epoch": epoch,
                    "sample_id": example.sample_id,
                    "target": example.target,
                    "prediction": example.prediction,
                    "cer": example.cer,
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as error:
        raise OcrEvaluationError("Не получилось сохранить примеры.") from error


def collect_matrix_labels(matrix: dict[str, dict[str, int]]) -> list[str]:
    labels = set(matrix)
    for columns in matrix.values():
        labels.update(columns)

    special_labels = [label for label in (DELETE_LABEL, INSERT_LABEL) if label in labels]
    normal_labels = sorted(labels - set(special_labels))
    return normal_labels + special_labels


def select_prediction_examples(
    examples: list[OcrPredictionExample],
    limit: int,
) -> list[OcrPredictionExample]:
    if limit <= 0:
        return []

    sorted_examples = sorted(examples, key=lambda item: item.cer, reverse=True)
    return sorted_examples[:limit]
