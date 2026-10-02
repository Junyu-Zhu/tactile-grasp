#!/usr/bin/env python3
"""Reversibly stage immutable token files in shared memory without path changes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


RAM_ROOT = Path("/dev/shm/force_slip_r4_tokens")
LOCK_PATH = Path("/dev/shm/force_slip_r4_tokens.lock")
BACKUP_SUFFIX = ".disk-original.npy"
CHUNK = 16 * 1024 * 1024
RESERVE = 1024 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def checked_manifest(cache_dir: Path, encoder: str) -> tuple[Path, dict[str, Any]]:
    cache_dir = cache_dir.resolve()
    path = (cache_dir / "cache_manifest.json").resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or payload.get("format") != "round4_htt_full_sparsh_tokens_v1":
        raise RuntimeError("RAM staging requires a complete Round 4 token cache")
    if payload.get("encoder") != encoder or len(payload.get("entries", [])) != 101:
        raise RuntimeError("Cache identity does not match requested encoder")
    episode_dir = (cache_dir / "episodes").resolve()
    originals = []
    names = []
    for entry in payload["entries"]:
        original = Path(os.path.abspath(entry["token_path"]))
        if not original.is_absolute() or original.parent.resolve() != episode_dir:
            raise RuntimeError(f"Token path escapes the cache episode directory: {original}")
        originals.append(str(original))
        names.append(original.name)
    if len(set(originals)) != 101 or len(set(names)) != 101:
        raise RuntimeError("Cache manifest token paths/names are not unique")
    return path, payload


def paths(entry: dict[str, Any], encoder: str) -> tuple[Path, Path, Path]:
    original = Path(entry["token_path"])
    if original.name != f"{original.stem.removesuffix('.tokens')}.tokens.npy":
        raise RuntimeError(f"Unexpected token filename: {original}")
    backup = original.with_name(original.name.removesuffix(".npy") + BACKUP_SUFFIX)
    ram = RAM_ROOT / encoder / original.name
    return original, backup, ram


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"dev": stat.st_dev, "ino": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def copy_verified(source: Path, target: Path, expected: str) -> dict[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    # The global flock permits one stable, managed temp name. A killed process
    # therefore leaves a file the next invocation can identify and remove.
    temp = target.with_name(target.name + ".staging-tmp")
    temp.unlink(missing_ok=True)
    before = file_identity(source)
    digest = hashlib.sha256()
    with source.open("rb") as src, temp.open("xb") as dst:
        for chunk in iter(lambda: src.read(CHUNK), b""):
            digest.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    after = file_identity(source)
    if before != after or digest.hexdigest() != expected:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Source changed while staging: {source}")
    os.replace(temp, target)
    if sha256_file(target) != expected:
        raise RuntimeError(f"RAM copy verification failed: {target}")
    return after


def verified_disk_identity(source: Path, expected: str) -> dict[str, int]:
    before = file_identity(source)
    if sha256_file(source) != expected:
        raise RuntimeError(f"Disk token hash mismatch: {source}")
    after = file_identity(source)
    if before != after:
        raise RuntimeError(f"Disk token changed during verification: {source}")
    return after


def stage_one(entry: dict[str, Any], encoder: str) -> dict[str, Any]:
    original, backup, ram = paths(entry, encoder)
    expected = entry["token_sha256"]
    if original.is_symlink():
        # A dangling RAM symlink after reboot is recoverable from the verified
        # disk backup; the link destination itself must still be the expected path.
        if original.absolute().readlink() != ram or not backup.is_file():
            raise RuntimeError(f"Invalid partial staged state: {original}")
        disk_identity = verified_disk_identity(backup, expected)
        if not ram.is_file() or sha256_file(ram) != expected:
            ram.unlink(missing_ok=True)
            copy_verified(backup, ram, expected)
        return {"original_path": str(original), "disk_backup_path": str(backup), "ram_path": str(ram),
                "sha256": expected, "bytes": disk_identity["size"],
                "disk_verified_identity": disk_identity, "status": "reused"}
    if backup.is_file() and original.is_file():
        if (backup.stat().st_dev, backup.stat().st_ino) != (original.stat().st_dev, original.stat().st_ino):
            raise RuntimeError(f"Ambiguous regular original plus backup: {original}")
    source = backup if backup.is_file() else original
    if not source.is_file():
        raise RuntimeError(f"Missing recoverable token source: {original}")
    if not ram.is_file() or sha256_file(ram) != expected:
        ram.unlink(missing_ok=True)
        disk_identity = copy_verified(source, ram, expected)
    else:
        disk_identity = verified_disk_identity(source, expected)
    if not backup.exists():
        # The hard link establishes the durable backup without ever removing the
        # manifest path. The following symlink replacement is one atomic rename.
        os.link(original, backup)
    elif file_identity(backup) != disk_identity:
        raise RuntimeError(f"Disk backup identity changed while staging: {backup}")
    temporary_link = original.with_name(original.name + ".ram-link-tmp")
    temporary_link.unlink(missing_ok=True)
    temporary_link.symlink_to(ram)
    os.replace(temporary_link, original)
    if not original.is_symlink() or original.resolve() != ram.resolve():
        raise RuntimeError(f"Atomic symlink activation failed: {original}")
    return {"original_path": str(original), "disk_backup_path": str(backup), "ram_path": str(ram),
            "sha256": expected, "bytes": disk_identity["size"],
            "disk_verified_identity": disk_identity, "status": "staged"}


def stage(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, manifest = checked_manifest(args.cache_dir, args.encoder)
    required = 0
    reclaimable = 0
    for entry in manifest["entries"]:
        original, backup, ram = paths(entry, args.encoder)
        source = backup if backup.is_file() else original
        ram_valid = ram.is_file() and sha256_file(ram) == entry["token_sha256"]
        if not ram_valid:
            if not source.is_file():
                raise RuntimeError(f"Missing recoverable token source: {original}")
            required += source.stat().st_size
            if ram.is_file():
                reclaimable += ram.stat().st_size
    free = shutil.disk_usage(RAM_ROOT.parent).free
    if required + RESERVE > free + reclaimable:
        raise RuntimeError(f"Insufficient shared memory: need {required + RESERVE}, free+reclaimable {free + reclaimable}")
    rows = []
    for index, entry in enumerate(manifest["entries"], 1):
        row = stage_one(entry, args.encoder)
        rows.append(row)
        print(f"[{index}/{len(manifest['entries'])}] {Path(row['original_path']).name}: {row['status']}", flush=True)
    payload = {
        "status": "staged",
        "encoder": args.encoder,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "ram_root": str(RAM_ROOT / args.encoder),
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
        # Each row is emitted only after its disk stream and RAM copy were
        # verified. Avoid another full HDD scan of all large token files.
        "all_hashes_match": len(rows) == len(manifest["entries"]),
    }
    atomic_json(args.record, payload)
    return payload


def restore(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, manifest = checked_manifest(args.cache_dir, args.encoder)
    record = json.loads(args.record.read_text())
    if record.get("status") != "staged" or record.get("encoder") != args.encoder:
        raise RuntimeError("Missing or incompatible staging record")
    if record.get("cache_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Cache manifest changed since staging")
    expected_paths = {}
    for entry in manifest["entries"]:
        original, backup, ram = paths(entry, args.encoder)
        expected_paths[str(original)] = {
            "disk_backup_path": str(backup), "ram_path": str(ram), "sha256": entry["token_sha256"]
        }
    by_original = {row.get("original_path"): row for row in record.get("files", [])}
    if set(by_original) != set(expected_paths) or len(record.get("files", [])) != len(expected_paths):
        raise RuntimeError("Staging record path set does not exactly match the manifest")
    for original_name, expected_row in expected_paths.items():
        row = by_original[original_name]
        if any(row.get(key) != value for key, value in expected_row.items()):
            raise RuntimeError(f"Staging record paths/hash were altered: {original_name}")
    encoder_dir = RAM_ROOT / args.encoder
    expected_ram = {Path(row["ram_path"]) for row in record["files"]}
    actual_ram = set(encoder_dir.iterdir()) if encoder_dir.is_dir() else set()
    if actual_ram != expected_ram:
        raise RuntimeError("RAM encoder directory contains missing or foreign content")
    # Preflight every path and hash before the first atomic restoration. Keep
    # the verified disk inode so the post-swap check needs no second HDD scan.
    disk_identities: dict[str, dict[str, int]] = {}
    for entry in manifest["entries"]:
        original, backup, ram = paths(entry, args.encoder)
        expected = entry["token_sha256"]
        if original.is_symlink():
            if original.absolute().readlink() != ram or not backup.is_file() or not ram.is_file():
                raise RuntimeError(f"Invalid staged link: {original}")
            disk_identities[str(original)] = verified_disk_identity(backup, expected)
            if sha256_file(ram) != expected:
                raise RuntimeError(f"Refusing restore after hash mismatch: {original}")
        elif original.is_file() and not backup.exists():
            disk_identities[str(original)] = verified_disk_identity(original, expected)
            if sha256_file(ram) != expected:
                raise RuntimeError(f"Refusing cleanup after RAM hash mismatch: {ram}")
        else:
            raise RuntimeError(f"Original is neither a valid staged link nor clean restored file: {original}")
    restored = []
    for entry in manifest["entries"]:
        original, backup, ram = paths(entry, args.encoder)
        expected = entry["token_sha256"]
        if original.is_symlink():
            os.replace(backup, original)
        if original.is_symlink() or file_identity(original) != disk_identities[str(original)]:
            raise RuntimeError(f"Disk restoration verification failed: {original}")
        restored.append(str(original))
    # Delete only RAM files enumerated by this verified record. Refuse hidden or
    # foreign content by requiring the encoder directory to become empty.
    for row in record["files"]:
        ram = Path(row["ram_path"])
        if ram.is_file() and sha256_file(ram) == row["sha256"]:
            ram.unlink()
        elif ram.exists():
            raise RuntimeError(f"Refusing to remove unverified RAM file: {ram}")
    encoder_dir.rmdir()
    if RAM_ROOT.exists() and not any(RAM_ROOT.iterdir()):
        RAM_ROOT.rmdir()
    payload = {**record, "status": "restored", "restored_files": restored,
               "ram_files_removed": len(record["files"])}
    atomic_json(args.record, payload)
    return payload


def status(args: argparse.Namespace) -> dict[str, Any]:
    _, manifest = checked_manifest(args.cache_dir, args.encoder)
    rows = []
    for entry in manifest["entries"]:
        original, backup, ram = paths(entry, args.encoder)
        rows.append({"original": str(original), "is_symlink": original.is_symlink(),
                     "backup_exists": backup.is_file(), "ram_exists": ram.is_file()})
    return {"encoder": args.encoder, "staged": sum(row["is_symlink"] for row in rows),
            "disk_backups": sum(row["backup_exists"] for row in rows),
            "ram_files": sum(row["ram_exists"] for row in rows), "files": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stage", "restore", "status"))
    parser.add_argument("--encoder", choices=("dino", "ijepa", "mae_letterbox"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = {"stage": stage, "restore": restore, "status": status}[args.command](args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
