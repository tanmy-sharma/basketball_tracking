"""TI-HOTA: line-faithful port of the official code (ti_hota_reference) and a vectorized version (ti_hota_fast).

Rows are (frame, x_cm, y_cm, attr) tuples, attr = team + jersey (e.g. "A10").
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 1.0, 0.05)
TAU = 100.0
LN05 = math.log(0.05)


# ----------------------------------------------------------------------------- reference
def _loc_sim(P, G, tau):
    d2 = np.sum((np.array(P) - np.array(G)) ** 2)
    return math.exp(math.log(0.05) * (d2 / (tau ** 2)))


def _group(rows):
    d: dict[int, list] = {}
    for f, x, y, a in rows:
        d.setdefault(int(f), []).append({"position": [x, y], "attributes": a})
    return d


def _alpha_ref(frames, tau, alpha):
    total_tp = total_fp = total_fn = 0
    assoc_counts: dict = {}
    fpa_counts: dict = {}
    for frame in frames:
        gt_objs, pred_objs = frame["ground_truths"], frame["predictions"]
        n_gt, n_pred = len(gt_objs), len(pred_objs)
        combined = np.zeros((n_gt, n_pred))
        cost = np.zeros((n_gt, n_pred))
        for i, gt in enumerate(gt_objs):
            for j, pred in enumerate(pred_objs):
                loc = _loc_sim(pred["position"], gt["position"], tau)
                combined[i, j] = loc * (1 if pred["attributes"] == gt["attributes"] else 0)
                cost[i, j] = 1 - combined[i, j]
        cost[combined < alpha] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        curr = {i: j for i, j in zip(row_ind, col_ind) if combined[i, j] >= alpha}
        tp = len(curr)
        total_tp += tp
        total_fp += n_pred - tp
        total_fn += n_gt - tp
        for i, gt in enumerate(gt_objs):
            assoc_counts.setdefault(gt["attributes"], {"TPA": 0, "FNA": 0})
            if i not in curr:
                assoc_counts[gt["attributes"]]["FNA"] += 1
        for i in curr:
            assoc_counts[gt_objs[i]["attributes"]]["TPA"] += 1
        for j, pred in enumerate(pred_objs):
            if j not in curr.values():
                fpa_counts[pred["attributes"]] = fpa_counts.get(pred["attributes"], 0) + 1
    total_det = total_tp + total_fp + total_fn
    det = total_tp / total_det if total_det > 0 else 0
    sum_A = n_tracks = 0
    for gid, cnt in assoc_counts.items():
        denom = cnt["TPA"] + cnt["FNA"] + fpa_counts.get(gid, 0)
        if denom > 0:
            sum_A += cnt["TPA"] / denom
            n_tracks += 1
    ass = sum_A / n_tracks if n_tracks > 0 else 0
    return math.sqrt(det * ass), det, ass


def _pack(ti, det, ass):
    ti, det, ass = map(np.asarray, (ti, det, ass))
    return {"TI-HOTA": float(ti.mean()), "DetA": float(det.mean()), "AssA": float(ass.mean()),
            "TI-HOTA_alphas": ti, "DetA_alphas": det, "AssA_alphas": ass}


def ti_hota_reference(gt_rows, pred_rows, tau: float = TAU) -> dict:
    g, p = _group(gt_rows), _group(pred_rows)
    frames = [{"ground_truths": g.get(f, []), "predictions": p.get(f, [])} for f in sorted(set(g) | set(p))]
    res = [_alpha_ref(frames, tau, a) for a in ALPHAS]
    return _pack(*zip(*res))


# ----------------------------------------------------------------------------- fast
def _arrays(rows):
    if not len(rows):
        return np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros(0, object)
    f, x, y, a = zip(*rows)
    return np.asarray(f, np.int64), np.c_[np.asarray(x, float), np.asarray(y, float)], np.asarray(a, object)


def ti_hota_fast(gt_rows, pred_rows, tau: float = TAU) -> dict:
    """Same result as the reference. Identity-gated similarity means a GT row can only match the
    prediction with the same attribute in the same frame, so (when attributes are unique per frame)
    matching is trivial and all alphas are handled by one threshold comparison. Frames with a
    duplicated attribute (in GT or pred) fall back to Hungarian."""
    gf, gp, ga = _arrays(gt_rows)
    pf, pp, pa = _arrays(pred_rows)
    names, inv = np.unique(np.concatenate([ga, pa]).astype(str), return_inverse=True)
    K, nA = len(names), len(ALPHAS)
    gc, pc = inv[: len(ga)], inv[len(ga):]
    nG, nP = np.bincount(gc, minlength=K), np.bincount(pc, minlength=K)

    gk, pk = gf * K + gc, pf * K + pc
    dup = set()
    for f, k in ((gf, gk), (pf, pk)):
        u, c = np.unique(k, return_counts=True)
        dup.update((u[c > 1] // K).tolist())
    dup_arr = np.fromiter(dup, np.int64, len(dup))
    gm, pm = ~np.isin(gf, dup_arr), ~np.isin(pf, dup_arr)

    TPA = np.zeros((K, nA))
    _, ig, ip = np.intersect1d(gk[gm], pk[pm], return_indices=True)
    if len(ig):
        gp_, pp_, codes = gp[gm][ig], pp[pm][ip], gc[gm][ig]
        s = np.exp(LN05 * ((gp_ - pp_) ** 2).sum(1) / tau ** 2)
        np.add.at(TPA, codes, s[:, None] >= ALPHAS[None, :])

    for f in dup:  # rare path: exact Hungarian per alpha
        gi, pi = np.flatnonzero(gf == f), np.flatnonzero(pf == f)
        d2 = ((gp[gi][:, None] - pp[pi][None]) ** 2).sum(-1)
        comb = np.exp(LN05 * d2 / tau ** 2) * (gc[gi][:, None] == pc[pi][None])
        for ai, alpha in enumerate(ALPHAS):
            cost = np.where(comb < alpha, 1e6, 1 - comb)
            for i, j in zip(*linear_sum_assignment(cost)):
                if comb[i, j] >= alpha:
                    TPA[gc[gi[i]], ai] += 1

    tp = TPA.sum(0)
    tot = len(gf) + len(pf) - tp  # TP + FP + FN
    det = np.divide(tp, tot, out=np.zeros(nA), where=tot > 0)
    m = nG > 0
    ass = (TPA[m] / (nG[m, None] + nP[m, None] - TPA[m])).mean(0) if m.any() else np.zeros(nA)
    return _pack(np.sqrt(det * ass), det, ass)
