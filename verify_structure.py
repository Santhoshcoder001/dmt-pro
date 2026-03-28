#!/usr/bin/env python3
"""Validate the required ABC-HP project structure.

Usage:
    python verify_structure.py
    python verify_structure.py --project-dir /path/to/abc_hp
    python verify_structure.py /path/to/abc_hp
"""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def colorize_status(text: str) -> str:
    if text == "FOUND":
        return f"{GREEN}{text}{RESET}"
    if text == "MISSING":
        return f"{RED}{text}{RESET}"
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify required folders and files for the ABC-HP project."
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        type=Path,
        default=None,
        help="Optional positional path to abc_hp project directory",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Path to the abc_hp project directory (default: ./abc_hp)",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path("check_report.txt"),
        help="Path to output report file (default: ./check_report.txt)",
    )
    return parser.parse_args()


def resolve_base_dir(args: argparse.Namespace) -> Path:
    if args.project_dir is not None:
        return args.project_dir
    if args.project_path is not None:
        return args.project_path
    return Path("abc_hp")


def check_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Return syntax check status for a Python file."""
    try:
        py_compile.compile(str(file_path), doraise=True)
        return True, "OK"
    except py_compile.PyCompileError as exc:
        return False, str(exc).strip()


def check_paths(base_dir: Path, report_file: Path) -> None:
    required_folders = [
        "data",
        "models",
        "api",
        "utils",
        "visualization",
    ]

    required_files = [
        "data/accident_data.csv",
        "models/model.pkl",
        "api/routes.py",
        "utils/data_loader.py",
        "utils/preprocessing.py",
        "utils/bias_correction.py",
        "utils/feature_engineering.py",
        "utils/hotspot_detection.py",
        "visualization/map.py",
        "config.py",
        "main.py",
        "requirements.txt",
    ]

    report_lines: list[str] = []

    def emit(message: str) -> None:
        print(message)
        report_lines.append(message)

    def emit_status(status: str, item_type: str, rel_path: str) -> None:
        emit(f"{colorize_status(status):8}  {item_type:6}  {rel_path}")

    emit("=" * 60)
    emit(f"Validating project structure at: {base_dir.resolve()}")
    emit("=" * 60)

    emit("\n[Folder Check]")
    missing_folders: list[str] = []
    for rel_path in required_folders:
        full_path = base_dir / rel_path
        exists = full_path.is_dir()
        status = "FOUND" if exists else "MISSING"
        emit_status(status, "folder", rel_path)
        if not exists:
            missing_folders.append(rel_path)

    emit("\n[File Check]")
    missing_files: list[str] = []
    empty_files: list[str] = []
    syntax_issues: list[tuple[str, str]] = []

    for rel_path in required_files:
        full_path = base_dir / rel_path
        exists = full_path.is_file()
        status = "FOUND" if exists else "MISSING"
        emit_status(status, "file", rel_path)

        if not exists:
            missing_files.append(rel_path)
            continue

        size = full_path.stat().st_size
        size_status = "FOUND" if size > 0 else "MISSING"
        emit_status(size_status, "size", f"{rel_path} ({size} bytes)")
        if size == 0:
            empty_files.append(rel_path)

        if full_path.suffix == ".py":
            valid_syntax, syntax_msg = check_python_syntax(full_path)
            syntax_status = "FOUND" if valid_syntax else "MISSING"
            emit_status(syntax_status, "syntax", rel_path)
            if not valid_syntax:
                syntax_issues.append((rel_path, syntax_msg))

    total_files_checked = len(required_files)
    total_missing_files = len(missing_files)

    emit("\n" + "-" * 60)
    emit("Summary")
    emit("-" * 60)
    emit(f"Total folders checked : {len(required_folders)}")
    emit(f"Total missing folders : {len(missing_folders)}")
    emit(f"Total files checked   : {total_files_checked}")
    emit(f"Total missing files   : {total_missing_files}")
    emit(f"Empty files found     : {len(empty_files)}")
    emit(f"Python syntax issues  : {len(syntax_issues)}")

    if missing_folders or missing_files or empty_files or syntax_issues:
        emit("\nSuggestions:")
        for folder in missing_folders:
            emit(f"- Create folder: {folder}")
        for file_name in missing_files:
            emit(f"- Create file:   {file_name}")
        for file_name in empty_files:
            emit(f"- Populate file (currently empty): {file_name}")
        for file_name, error in syntax_issues:
            emit(f"- Fix syntax in {file_name}: {error}")
    else:
        emit("\nAll required files and folders are present, non-empty, and valid.")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    sanitized_report = "\n".join(line.replace(GREEN, "").replace(RED, "").replace(RESET, "") for line in report_lines)
    report_file.write_text(sanitized_report + "\n", encoding="utf-8")
    emit(f"\nReport written to: {report_file.resolve()}")

    # Optional auto-create logic (commented on purpose).
    # Uncomment this block only if you want to create missing paths automatically.
    #
    # for folder in missing_folders:
    #     (base_dir / folder).mkdir(parents=True, exist_ok=True)
    #
    # for file_name in missing_files:
    #     target = base_dir / file_name
    #     target.parent.mkdir(parents=True, exist_ok=True)
    #     target.touch(exist_ok=True)


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args)
    check_paths(base_dir, args.report_file)


if __name__ == "__main__":
    main()
