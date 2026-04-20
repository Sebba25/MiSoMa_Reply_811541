"""Cache + output path conventions for the cleaning pipeline.

All paths live under ``<dataset.parent>/.cleaning_cache/<dataset.stem>/``.
Helpers for the generated-cleaners directory, the cleaned CSV, the cleaner
manifest, and the final report are centralised here so stage code never
hard-codes paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from models import ColumnCleanerProgram, GeneratedCleanerArtifact
from tools import normalized_schema_name


def cleaning_cache_dir(path: Path) -> Path:
    return path.parent / ".cleaning_cache" / path.stem


def generated_cleaner_path(path: Path, column_name: str) -> Path:
    return cleaning_cache_dir(path) / "generated_cleaners" / f"{normalized_schema_name(column_name)}.py"


def cleaned_dataset_path(path: Path) -> Path:
    return cleaning_cache_dir(path) / f"{path.stem}.cleaned.csv"


def cleaner_manifest_path(path: Path) -> Path:
    return cleaning_cache_dir(path) / "cleaner_manifest.json"


def final_report_path(path: Path) -> Path:
    return cleaning_cache_dir(path) / f"{path.stem}.final_report.json"


def save_cleaner_manifest(path: Path, artifacts: list[GeneratedCleanerArtifact]) -> None:
    manifest_path = cleaner_manifest_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([artifact.model_dump() for artifact in artifacts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_cleaner_manifest(path: Path) -> list[GeneratedCleanerArtifact]:
    manifest_path = cleaner_manifest_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cleaner manifest not found at {manifest_path}. Run --stage generate first."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [GeneratedCleanerArtifact.model_validate(entry) for entry in data]


def save_generated_cleaner(path: Path, program: ColumnCleanerProgram) -> Path:
    code_path = generated_cleaner_path(path, program.column_name)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(program.python_code.strip() + "\n", encoding="utf-8")
    return code_path

