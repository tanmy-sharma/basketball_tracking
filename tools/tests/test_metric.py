import numpy as np
import pytest

from trackid import io
from trackid.metric import ti_hota_fast, ti_hota_reference

N = 600


@pytest.fixture(scope="module")
def gt():
    df = io.load_court_gt("IMG_0104")
    return io.gt_rows(df[df.frame <= N])


def test_perfect(gt):
    for fn in (ti_hota_fast, ti_hota_reference):
        r = fn(gt, gt)
        assert r["TI-HOTA"] == pytest.approx(1.0)


def test_empty(gt):
    for fn in (ti_hota_fast, ti_hota_reference):
        assert fn(gt, [])["TI-HOTA"] == 0
        assert fn([], gt)["TI-HOTA"] == 0


def _agree(gt, pred):
    a, b = ti_hota_fast(gt, pred), ti_hota_reference(gt, pred)
    for k in ("TI-HOTA_alphas", "DetA_alphas", "AssA_alphas"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-12)
    assert a["TI-HOTA"] == pytest.approx(b["TI-HOTA"])


@pytest.mark.parametrize("seed", range(4))
def test_fast_matches_reference(gt, seed):
    rng = np.random.default_rng(seed)
    pred = []
    attrs = sorted({r[3] for r in gt})
    for f, x, y, a in gt:
        u = rng.random()
        if u < 0.05:
            continue  # drop
        if u < 0.10:
            a = attrs[rng.integers(len(attrs))]  # wrong label (may create duplicates)
        pred.append((f, x + rng.normal(0, 40), y + rng.normal(0, 40), a))
        if rng.random() < 0.03:  # spurious / duplicate-identity row
            pred.append((f, x + rng.normal(0, 80), y + rng.normal(0, 80), a if seed % 2 else "X0"))
    _agree(gt, pred)
