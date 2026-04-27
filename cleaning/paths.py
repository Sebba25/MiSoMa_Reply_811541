"""Cache + output path conventions for the cleaning pipeline.

All paths live under ``<dataset.parent>/.cleaning_cache/<dataset.stem>/``.
Helpers for the generated-cleaners directory, the cleaned CSV, the cleaner
manifest, and the final report are centralised here so stage code never
hard-codes paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.models import ColumnCleanerProgram, GeneratedCleanerArtifact
from tools import normalized_schema_name 


def cleaning_cache_dir(path: Path) -> Path:
    '''Builds the root cache directory used by the cleaning pipeline for one dataset.'''
    # Store all cleaning outputs under <dataset.parent>/.cleaning_cache/<dataset.stem>/
    return path.parent / ".cleaning_cache" / path.stem
#It gives one standard root folder for all cleaning outputs related to one dataset

def generated_cleaner_path(path: Path, column_name: str) -> Path:
    '''Builds the file path of the generated cleaner Python file for one column.'''
    # Normalize the column name so it becomes a safe and consistent filename
    return cleaning_cache_dir(path) / "generated_cleaners" / f"{normalized_schema_name(column_name)}.py"
#Each generated cleaner needs a stable and safe filename based on its column

def cleaned_dataset_path(path: Path) -> Path:
    '''Builds the output path of the cleaned CSV produced by the application stage.'''
    return cleaning_cache_dir(path) / f"{path.stem}.cleaned.csv"
#It gives one standard output location for the cleaned dataset

def cleaner_manifest_path(path: Path) -> Path:
    '''Builds the path of the JSON manifest that lists all generated cleaner artifacts.'''
    return cleaning_cache_dir(path) / "cleaner_manifest.json"
#The cleaner manifest stores the list of generated cleaner artifacts for the dataset

def final_report_path(path: Path) -> Path:
    '''Builds the path of the final structured JSON report for the dataset.'''
    return cleaning_cache_dir(path) / f"{path.stem}.final_report.json"
#It gives one standard location for the final structured pipeline report

def save_cleaner_manifest(path: Path, artifacts: list[GeneratedCleanerArtifact]) -> None:
    '''Saves the cleaner artifact manifest as a JSON file.'''
    #it takes the dataset path as wel as artifacts: the list of generated cleaner artifacts
    # Compute the standard manifest path inside the cleaning cache directory
    manifest_path = cleaner_manifest_path(path)
    # Make sure the parent folder exists before writing the file
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert each artifact model into plain data and save the whole list as JSON
    #artifact.model_dump() converts each artifact model into a plain dictionary
    #the list comprehension collects all artifact dictionaries
    #json.dumps(...) converts that list into formatted JSON text
    #indent=2 makes it pretty-printed
    #ensure_ascii=False allows non-ASCII characters to stay readable
    manifest_path.write_text(
        json.dumps([artifact.model_dump() for artifact in artifacts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
#It stores a reusable manifest describing all generated cleaners

def load_cleaner_manifest(path: Path) -> list[GeneratedCleanerArtifact]:
    '''Loads the cleaner artifact manifest from disk and validates each entry as a model.'''
    # Compute the expected manifest location
    manifest_path = cleaner_manifest_path(path)
    if not manifest_path.exists():
        # Raise a clear error if generation has not been run yet
        raise FileNotFoundError(
            f"Cleaner manifest not found at {manifest_path}. Run --stage generate first."
        )
    # Read the JSON file and parse it into plain Python data
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Convert each saved entry back into a validated GeneratedCleanerArtifact model
    return [GeneratedCleanerArtifact.model_validate(entry) for entry in data]
#It reconstructs the saved cleaner artifact records from disk

def save_generated_cleaner(path: Path, program: ColumnCleanerProgram) -> Path:
    '''Saves one generated cleaner program as a Python file and returns its path.'''
    # Build the cleaner file path from the dataset path and the program's column name
    code_path = generated_cleaner_path(path, program.column_name)
    # Ensure the generated_cleaners directory exists before writing the file
    code_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the generated Python code, trimming outer whitespace and ending with one newline
    #program.python_code is the generated source code
    #.strip() removes extra spaces/newlines at the beginning and end
    code_path.write_text(program.python_code.strip() + "\n", encoding="utf-8")
    #it returns the saved file path
    return code_path
#It saves each generated cleaner as a real Python file that can later be loaded and executed
