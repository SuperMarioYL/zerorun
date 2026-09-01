"""Lockstep version-surface test (v0.4.0 fix-init-version-drift).

The shipped v0.3.0 release mis-reported its own version: ``__version__`` in
``src/zerorun/__init__.py`` was ``0.1.0`` — two minor versions behind the
canonical ``VERSION`` file (``0.3.0``) and ``pyproject.toml`` (``0.3.0``) —
because the v0.2.0 and v0.3.0 bumps each touched only ``VERSION`` +
``pyproject.toml`` + ``CHANGELOG.md`` and never propagated to
``__init__.py.__version__``. This test asserts every version surface agrees
with the canonical ``VERSION`` file so the next bump cannot silently leave
``__version__`` stale — the exact recurrence that produced the drift.

The ``VERSION`` file is the single source of truth (it is the surface the
release process bumps first). ``pyproject.toml``'s ``version`` and
``zerorun.__version__`` must each equal it. The test does not hardcode the
version number, so it auto-tracks each bump and only fails on partial drift.
"""

from __future__ import annotations

import re
import sys
import tomllib
import unittest
from pathlib import Path

# Make src/ importable when tests run without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import zerorun  # noqa: E402

# A loose semver shape (leading ``v`` optional) so a blank/garbage VERSION is
# also caught, not just inter-surface disagreement.
_SEMVER_RE = re.compile(r"^v?\d+\.\d+\.\d+$")


class TestVersionLockstep(unittest.TestCase):
    """Every shipped version surface must agree with the canonical VERSION file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.version_file_path = ROOT / "VERSION"
        cls.pyproject_path = ROOT / "pyproject.toml"

    def test_version_file_exists_and_is_semver(self) -> None:
        self.assertTrue(
            self.version_file_path.exists(),
            f"canonical VERSION file missing: {self.version_file_path}",
        )
        raw = self.version_file_path.read_text(encoding="utf-8").strip()
        self.assertTrue(
            _SEMVER_RE.match(raw),
            f"VERSION file is not a clean semver: {raw!r}",
        )
        cls_version = raw[1:] if raw.startswith("v") else raw
        self.assertEqual(raw, cls_version, msg="VERSION file carries no 'v' prefix")

    def test_pyproject_version_matches_version_file(self) -> None:
        file_ver = self.version_file_path.read_text(encoding="utf-8").strip()
        with open(self.pyproject_path, "rb") as fh:
            data = tomllib.load(fh)
        pyproject_ver = data["project"]["version"]
        self.assertEqual(
            pyproject_ver, file_ver,
            msg=(
                "pyproject.toml 'version' drifts from the VERSION file — a bump "
                "touched only one surface (the version-drift recurrence)."
            ),
        )

    def test_dunder_version_matches_version_file(self) -> None:
        file_ver = self.version_file_path.read_text(encoding="utf-8").strip()
        self.assertEqual(
            zerorun.__version__, file_ver,
            msg=(
                f"zerorun.__version__ ({zerorun.__version__!r}) drifts from the "
                f"VERSION file ({file_ver!r}) — a bump touched VERSION + "
                "pyproject.toml but never propagated to __init__.py.__version__ "
                "(the v0.2.0/v0.3.0 recurrence this test exists to lock out)."
            ),
        )

    def test_all_three_surfaces_agree(self) -> None:
        file_ver = self.version_file_path.read_text(encoding="utf-8").strip()
        with open(self.pyproject_path, "rb") as fh:
            pyproject_ver = tomllib.load(fh)["project"]["version"]
        surfaces = {
            "VERSION file": file_ver,
            "pyproject.toml": pyproject_ver,
            "zerorun.__version__": zerorun.__version__,
        }
        unique = set(surfaces.values())
        self.assertEqual(
            len(unique), 1,
            msg=f"version surfaces disagree: {surfaces}",
        )
        (sole,) = unique
        self.assertEqual(sole, file_ver)


if __name__ == "__main__":
    unittest.main()
