from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _outputs(tmp_path, monkeypatch):
    """Every test writes runs/crops under its own tmp dir, never the repo's outputs/."""
    monkeypatch.setenv("SRC_OUTPUTS", str(tmp_path / "outputs"))
