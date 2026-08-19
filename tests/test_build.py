from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paper_build_sets_reproducible_pdf_environment(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    build_log = tmp_path / "pdflatex-environment.txt"

    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    fake_pdflatex = fake_bin / "pdflatex"
    fake_pdflatex.write_text(
        '#!/bin/sh\nprintf "%s,%s\\n" "$SOURCE_DATE_EPOCH" '
        '"$FORCE_SOURCE_DATE" >> "$BUILD_LOG"\n',
        encoding="utf-8",
    )
    fake_pdflatex.chmod(0o755)

    fake_bibtex = fake_bin / "bibtex"
    fake_bibtex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bibtex.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["BUILD_LOG"] = str(build_log)

    subprocess.run(
        ["make", "paper", "PYTHON=python"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert build_log.read_text(encoding="utf-8").splitlines() == ["0,1", "0,1", "0,1"]
