import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from trackid import io
from validate_submission import check_file

TOOLS = Path(__file__).resolve().parents[1]
ROSTERS = io.load_roster(io.FINAL / "games_roster.csv")
IDS = ["A,6", "A,9", "A,10", "B,6", "B,9", "B,11"]  # IMG_0114 roster


def sub(n_frames: int, coord: str = "100.0") -> str:
    return "".join(f"{f},{coord},200.0,{i}\n" for f in range(1, n_frames + 1) for i in IDS)


def run_check(text: str, n: int = 3):
    err, warn = [], []
    check_file("IMG_0114.txt", text, n, ROSTERS["IMG_0114"], err, warn)
    return err, warn


def test_valid_passes():
    err, warn = run_check(sub(3))
    assert err == [] and warn == []


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_non_finite_rejected(bad):
    err, _ = run_check(sub(3, bad))
    assert any("non-finite" in e for e in err)


def test_duplicate_zip_entries_rejected(tmp_path):
    good, bad = sub(3), "not,a,valid,file\n"
    zp = tmp_path / "dup.zip"
    with zipfile.ZipFile(zp, "w") as z, pytest.warns(UserWarning):
        z.writestr("IMG_0114.txt", bad)
        z.writestr("IMG_0114.txt", good)
        z.writestr("IMG_0115.txt", good)
    r = subprocess.run([sys.executable, "validate_submission.py", str(zp), "--frames", "IMG_0114=3", "IMG_0115=3"],
                       cwd=TOOLS, capture_output=True, text=True)
    assert r.returncode == 1
    assert "2 entries named IMG_0114.txt" in r.stdout
