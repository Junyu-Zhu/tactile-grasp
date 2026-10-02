#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import ram_cache


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RamCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.cache = root / "cache"
        episodes = self.cache / "episodes"
        episodes.mkdir(parents=True)
        self.ram = root / "ram"
        self.old_ram_root = ram_cache.RAM_ROOT
        ram_cache.RAM_ROOT = self.ram
        entries = []
        for index in range(101):
            data = f"token-{index}".encode()
            path = episodes / f"episode-{index:03d}.tokens.npy"
            path.write_bytes(data)
            entries.append({"token_path": str(path), "token_sha256": digest(data)})
        (self.cache / "cache_manifest.json").write_text(json.dumps({
            "status": "complete", "format": "round4_htt_full_sparsh_tokens_v1",
            "encoder": "dino", "entries": entries,
        }))
        self.record = root / "record.json"
        self.args = argparse.Namespace(cache_dir=self.cache, encoder="dino", record=self.record)

    def tearDown(self) -> None:
        ram_cache.RAM_ROOT = self.old_ram_root
        self.temp.cleanup()

    def test_stage_reuse_and_restore_are_byte_exact(self) -> None:
        staged = ram_cache.stage(self.args)
        self.assertTrue(staged["all_hashes_match"])
        first = Path(staged["files"][0]["original_path"])
        backup = Path(staged["files"][0]["disk_backup_path"])
        self.assertTrue(first.is_symlink())
        self.assertTrue(backup.is_file())
        reused = ram_cache.stage(self.args)
        self.assertEqual({row["status"] for row in reused["files"]}, {"reused"})
        restored = ram_cache.restore(self.args)
        self.assertEqual(restored["status"], "restored")
        self.assertFalse(first.is_symlink())
        self.assertEqual(digest(first.read_bytes()), staged["files"][0]["sha256"])
        self.assertFalse(backup.exists())
        self.assertFalse((self.ram / "dino").exists())

    def test_recovers_missing_original_from_backup(self) -> None:
        entry = json.loads((self.cache / "cache_manifest.json").read_text())["entries"][0]
        original, backup, ram = ram_cache.paths(entry, "dino")
        original.rename(backup)
        row = ram_cache.stage_one(entry, "dino")
        self.assertEqual(row["status"], "staged")
        self.assertTrue(original.is_symlink())
        self.assertTrue(backup.is_file())
        self.assertTrue(ram.is_file())

    def test_recovers_broken_ram_symlink(self) -> None:
        ram_cache.stage(self.args)
        entry = json.loads((self.cache / "cache_manifest.json").read_text())["entries"][0]
        original, _, ram = ram_cache.paths(entry, "dino")
        ram.unlink()
        self.assertTrue(original.is_symlink())
        row = ram_cache.stage_one(entry, "dino")
        self.assertEqual(row["status"], "reused")
        self.assertTrue(ram.is_file())

    def test_accepts_hardlink_crash_state(self) -> None:
        entry = json.loads((self.cache / "cache_manifest.json").read_text())["entries"][0]
        original, backup, _ = ram_cache.paths(entry, "dino")
        os.link(original, backup)
        row = ram_cache.stage_one(entry, "dino")
        self.assertEqual(row["status"], "staged")
        self.assertTrue(original.is_symlink())

    def test_tampered_record_is_rejected_before_restore(self) -> None:
        staged = ram_cache.stage(self.args)
        first = Path(staged["files"][0]["original_path"])
        staged["files"][0]["ram_path"] = str(Path(self.temp.name) / "foreign")
        self.record.write_text(json.dumps(staged))
        with self.assertRaisesRegex(RuntimeError, "altered"):
            ram_cache.restore(self.args)
        self.assertTrue(first.is_symlink())

    def test_stale_managed_temps_are_recovered(self) -> None:
        manifest = json.loads((self.cache / "cache_manifest.json").read_text())
        entry = manifest["entries"][0]
        original, _, ram = ram_cache.paths(entry, "dino")
        ram.parent.mkdir(parents=True)
        ram.with_name(ram.name + ".staging-tmp").write_bytes(b"partial")
        stale_link = original.with_name(original.name + ".ram-link-tmp")
        stale_link.symlink_to(ram)
        ram_cache.stage(self.args)
        self.assertFalse(ram.with_name(ram.name + ".staging-tmp").exists())
        self.assertFalse(stale_link.exists())
        ram_cache.restore(self.args)
        self.assertFalse((self.ram / "dino").exists())

    def test_foreign_ram_content_is_rejected_before_restore(self) -> None:
        staged = ram_cache.stage(self.args)
        first = Path(staged["files"][0]["original_path"])
        (self.ram / "dino" / "foreign").write_bytes(b"foreign")
        with self.assertRaisesRegex(RuntimeError, "foreign content"):
            ram_cache.restore(self.args)
        self.assertTrue(first.is_symlink())

    def test_manifest_escape_is_rejected(self) -> None:
        manifest_path = self.cache / "cache_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["entries"][0]["token_path"] = str(Path(self.temp.name) / "escape.tokens.npy")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            ram_cache.checked_manifest(self.cache, "dino")


if __name__ == "__main__":
    unittest.main()
