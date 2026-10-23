"""Release/install-surface guards (v0.6.0).

Two shipped surfaces failed silently and are pinned here so they cannot regress
without a red suite:

* ``.github/workflows/release.yml`` built ``dist/`` but only uploaded it as an
  Actions workflow artifact, and its PyPI job is gated off — so every release
  (v0.1.0–v0.5.0) carried zero downloadable assets even though the release runs
  were green. The workflow must attach the built files to the GitHub Release.
* ``examples/quickstart.sh`` opened with ``uv tool install zerorun``, which
  cannot resolve (the distribution is not on PyPI), so the shipped example
  failed on its first line. It must use the checkout install both READMEs
  document.

These are structural assertions over the committed files (the workflow only runs
on a tag push and the example needs a real environment), so they check the
contract each surface has to satisfy rather than executing it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
QUICKSTART = ROOT / "examples" / "quickstart.sh"


def _steps(job: dict) -> list[dict]:
    return [s for s in job.get("steps", []) if isinstance(s, dict)]


def _run_texts(job: dict) -> list[str]:
    return [s.get("run", "") for s in _steps(job) if isinstance(s.get("run"), str)]


class TestReleaseWorkflowAttachesAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
        cls.build = cls.doc["jobs"]["build"]

    def test_workflow_builds_dist(self) -> None:
        runs = _run_texts(self.build)
        self.assertTrue(
            any("python -m build" in r for r in runs),
            msg="the build job must still produce dist/",
        )

    def test_release_attach_step_present_and_uploaded(self) -> None:
        runs = _run_texts(self.build)
        attach = [r for r in runs if "gh release upload" in r]
        self.assertTrue(
            attach,
            msg=(
                "release.yml must attach dist/ to the GitHub Release — without a "
                "gh release upload step the release carries no wheel/sdist"
            ),
        )
        text = "\n".join(attach)
        self.assertIn("dist/*", text)
        self.assertIn("$GITHUB_REF_NAME", text)
        self.assertIn(
            "--clobber", text,
            msg=(
                "the upload must be idempotent (--clobber) because the tag push "
                "can precede the release creation and a local dist/ publish can "
                "upload the same asset names"
            ),
        )

    def test_attach_step_creates_the_release_if_missing(self) -> None:
        # publish_repo.sh pushes the tag before it creates the release, so the
        # workflow must tolerate either order.
        runs = _run_texts(self.build)
        text = "\n".join(r for r in runs if "gh release upload" in r)
        self.assertIn("gh release view", text)
        self.assertIn("gh release create", text)

    def test_workflow_can_write_releases(self) -> None:
        self.assertEqual(self.doc.get("permissions", {}).get("contents"), "write")

    def test_pypi_job_stays_gated_off(self) -> None:
        # The release-asset step must not depend on PyPI publishing.
        self.assertIn("pypi", self.doc["jobs"])
        self.assertNotIn(
            "gh release upload",
            "\n".join(_run_texts(self.doc["jobs"]["pypi"])),
        )


class TestQuickstartInstallCommand(unittest.TestCase):
    def test_no_unresolvable_pypi_install(self) -> None:
        # Only executable lines count — the header comment explains why the PyPI
        # install is not used, and must be free to name it.
        commands = [
            line.strip()
            for line in QUICKSTART.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        offenders = [c for c in commands if c.startswith("uv tool install")]
        self.assertEqual(
            offenders, [],
            msg=(
                "zerorun is not published to PyPI, so `uv tool install zerorun` "
                f"cannot resolve; the example must install from a checkout: {offenders}"
            ),
        )

    def test_installs_from_the_checkout(self) -> None:
        text = QUICKSTART.read_text(encoding="utf-8")
        self.assertIn("uv venv .venv", text)
        self.assertIn("uv pip install --python .venv/bin/python -e .", text)
        self.assertIn("git clone https://github.com/SuperMarioYL/zerorun.git", text)

    def test_demo_commands_use_the_installed_binary(self) -> None:
        text = QUICKSTART.read_text(encoding="utf-8")
        for cmd in ("probe --gpu", "deepseek-v4 --dry-run --gpu", "--list --gpu"):
            self.assertIn(
                f".venv/bin/zerorun {cmd}", text,
                msg=f"the example must drive the installed binary for {cmd!r}",
            )


if __name__ == "__main__":
    unittest.main()
