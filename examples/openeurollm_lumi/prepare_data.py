"""Convert frozen OpenEuroLLM math pools to verl's native prompt/reward schema."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def select_eu(candidates: list[dict], count: int) -> list[dict]:
    by_language: dict[str, list[dict]] = {}
    for row in candidates:
        by_language.setdefault(row["language"], []).append(row)
    for language in by_language:
        by_language[language].sort(key=lambda row: row["id"])
    languages = sorted(by_language)
    result = []
    while len(result) < count:
        for language in languages:
            if by_language[language] and len(result) < count:
                result.append(by_language[language].pop(0))
        if not any(by_language.values()):
            raise ValueError("not enough EU rows")
    return result


def convert(row: dict, split: str) -> dict:
    if row["verifier_kind"] not in {"integer_exact", "rational_exact"}:
        raise ValueError(f"unsupported verifier: {row['verifier_kind']}")
    return {
        "data_source": "oellm_math_rlvr",
        "prompt": row["messages"],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": row["ground_truth"]},
        "extra_info": {
            "split": split,
            "source_id": row["id"],
            "language": row["language"],
            "semantic_group_id": row["semantic_group_id"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eu-train", type=Path, required=True)
    parser.add_argument("--en-train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.parquet"
    val_path = output / "validation.parquet"
    if train_path.exists() or val_path.exists():
        raise FileExistsError("refusing to overwrite a frozen verl dataset")

    eu = rows(args.eu_train)
    en = sorted(rows(args.en_train), key=lambda row: row["id"])
    holdout = rows(args.holdout)
    selected_eu = select_eu(eu, 13)
    # With an eight-prompt training batch and no shuffle, the first update
    # sees six EU prompts and two English replay prompts.
    train = selected_eu[:6] + en[:2] + selected_eu[6:] + en[2:3]
    validation = select_eu(holdout, 16)
    train_groups = {row["semantic_group_id"] for row in train}
    val_groups = {row["semantic_group_id"] for row in validation}
    if train_groups & val_groups:
        raise ValueError("semantic groups overlap between train and validation")
    if len(train_groups) != len(train) or len(val_groups) != len(validation):
        raise ValueError("duplicate semantic groups inside a split")

    pq.write_table(pa.Table.from_pylist([convert(row, "train") for row in train]), train_path)
    pq.write_table(pa.Table.from_pylist([convert(row, "validation") for row in validation]), val_path)
    for source in (args.eu_train, args.en_train, args.holdout, train_path, val_path):
        print(f"sha256 {sha256(source)} {source}")
    print("train_languages", dict(sorted(Counter(row["language"] for row in train).items())))
    print("validation_languages", dict(sorted(Counter(row["language"] for row in validation).items())))


if __name__ == "__main__":
    main()
