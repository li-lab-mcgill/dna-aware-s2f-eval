#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from path_config import DATASET_DIR, PROJECT_DIR, TRAINED_CHECKPOINT_DIR, WORKSPACE_DIR


DEFAULT_MANIFEST_PATH = PROJECT_DIR / "zenodo_manifest.json"
REQUIRED_DATASET_FILES = ("peaks.all_folds.zarr", "nonpeaks.fold_0.zarr")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download paired dataset and checkpoint bundles from Zenodo into the repo workspace."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the Zenodo manifest JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Subset of dataset IDs to download. Defaults to all bundles in the manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and replace any existing extracted bundle directories and readmes.",
    )
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open() as handle:
        manifest = json.load(handle)

    if "zenodo_record" not in manifest:
        raise KeyError(f"Manifest is missing 'zenodo_record': {manifest_path}")
    if "bundles" not in manifest or not isinstance(manifest["bundles"], dict):
        raise KeyError(f"Manifest is missing a 'bundles' mapping: {manifest_path}")
    if "readmes" not in manifest or not isinstance(manifest["readmes"], dict):
        raise KeyError(f"Manifest is missing a 'readmes' mapping: {manifest_path}")
    return manifest


def _resolve_selection(manifest: dict, requested: list[str] | None) -> list[str]:
    available = sorted(manifest["bundles"])
    if not requested:
        return available

    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(
            "Unknown dataset ID(s): "
            f"{', '.join(invalid)}. Supported dataset IDs: {', '.join(available)}"
        )
    return requested


def _safe_extract(zip_path: Path, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (target_dir / member.filename).resolve()
            if target_root not in (member_path, *member_path.parents):
                raise RuntimeError(f"Unsafe zip member path in {zip_path}: {member.filename}")
        archive.extractall(target_dir)


def _download_archive(*, zenodo_record: str, archive_name: str, output_dir: Path, force: bool) -> Path:
    try:
        from zenodo_get import download as zenodo_download
    except ImportError as exc:
        raise RuntimeError(
            "zenodo-get is not installed in the active environment. "
            "Run `./init_env.sh` and then `conda activate cglm`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name

    if archive_path.exists() and force:
        archive_path.unlink()

    if archive_path.exists():
        return archive_path

    zenodo_download(
        record_or_doi=zenodo_record,
        output_dir=output_dir,
        file_glob=[archive_name],
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"Zenodo download completed without creating {archive_path}")
    return archive_path


def _dataset_ready(dataset_id: str) -> bool:
    dataset_root = DATASET_DIR / dataset_id
    return dataset_root.is_dir() and all((dataset_root / name).exists() for name in REQUIRED_DATASET_FILES)


def _checkpoint_ready(dataset_id: str) -> bool:
    checkpoint_root = TRAINED_CHECKPOINT_DIR / f"{dataset_id}___checkpoints"
    return checkpoint_root.is_dir() and any(checkpoint_root.iterdir())


def _prepare_dataset_bundle(
    dataset_id: str,
    archive_name: str,
    zenodo_record: str,
    *,
    force: bool,
    downloads_root: Path,
) -> None:
    dataset_root = DATASET_DIR / dataset_id

    if _dataset_ready(dataset_id) and not force:
        print(f"Skipping dataset bundle for {dataset_id}: already prepared at {dataset_root}")
        return

    if dataset_root.exists() and not force:
        raise RuntimeError(
            f"Dataset directory already exists but is incomplete: {dataset_root}. "
            "Re-run with --force to replace it."
        )

    archive_path = _download_archive(
        zenodo_record=zenodo_record,
        archive_name=archive_name,
        output_dir=downloads_root,
        force=force,
    )

    if force and dataset_root.exists():
        print(f"Removing existing dataset directory for {dataset_id}")
        shutil.rmtree(dataset_root)

    print(f"Extracting {archive_path.name} into {DATASET_DIR}")
    _safe_extract(archive_path, DATASET_DIR)

    if not _dataset_ready(dataset_id):
        raise RuntimeError(
            f"Extraction for {dataset_id} did not produce the expected dataset layout under {dataset_root}"
        )


def _prepare_checkpoint_bundle(
    dataset_id: str,
    archive_name: str,
    zenodo_record: str,
    *,
    force: bool,
    downloads_root: Path,
) -> None:
    checkpoint_root = TRAINED_CHECKPOINT_DIR / f"{dataset_id}___checkpoints"

    if _checkpoint_ready(dataset_id) and not force:
        print(f"Skipping checkpoint bundle for {dataset_id}: already prepared at {checkpoint_root}")
        return

    if checkpoint_root.exists() and not force:
        raise RuntimeError(
            f"Checkpoint directory already exists but is incomplete: {checkpoint_root}. "
            "Re-run with --force to replace it."
        )

    archive_path = _download_archive(
        zenodo_record=zenodo_record,
        archive_name=archive_name,
        output_dir=downloads_root,
        force=force,
    )

    if force and checkpoint_root.exists():
        print(f"Removing existing checkpoint directory for {dataset_id}")
        shutil.rmtree(checkpoint_root)

    print(f"Extracting {archive_path.name} into {TRAINED_CHECKPOINT_DIR}")
    _safe_extract(archive_path, TRAINED_CHECKPOINT_DIR)

    if not _checkpoint_ready(dataset_id):
        raise RuntimeError(
            f"Extraction for {dataset_id} did not produce the expected checkpoint layout under {checkpoint_root}"
        )


def _prepare_readmes(manifest: dict, *, force: bool, downloads_root: Path) -> None:
    zenodo_record = str(manifest["zenodo_record"])

    for readme_cfg in manifest["readmes"].values():
        archive_name = readme_cfg["archive_name"]
        destination_dir = PROJECT_DIR / readme_cfg["destination_dir"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / archive_name

        if destination_path.exists() and not force:
            continue

        archive_path = _download_archive(
            zenodo_record=zenodo_record,
            archive_name=archive_name,
            output_dir=downloads_root,
            force=force,
        )
        shutil.copy2(archive_path, destination_path)
        print(f"Prepared {archive_name} at {destination_path}")


def _cleanup_downloads(downloads_root: Path) -> None:
    if downloads_root.exists():
        shutil.rmtree(downloads_root)


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    dataset_ids = _resolve_selection(manifest, args.datasets)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    downloads_root = WORKSPACE_DIR / ".downloads" / "zenodo"
    if args.force and downloads_root.exists():
        shutil.rmtree(downloads_root)

    try:
        for dataset_id in dataset_ids:
            bundle_cfg = manifest["bundles"][dataset_id]
            _prepare_dataset_bundle(
                dataset_id,
                bundle_cfg["dataset_archive"],
                str(manifest["zenodo_record"]),
                force=args.force,
                downloads_root=downloads_root,
            )
            _prepare_checkpoint_bundle(
                dataset_id,
                bundle_cfg["checkpoint_archive"],
                str(manifest["zenodo_record"]),
                force=args.force,
                downloads_root=downloads_root,
            )

        if dataset_ids:
            _prepare_readmes(manifest, force=args.force, downloads_root=downloads_root)
    finally:
        _cleanup_downloads(downloads_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
