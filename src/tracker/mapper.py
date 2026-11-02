"""Maps box foot points to the court in metres, with their uncertainty. The tracker's defaults assume metres."""
from __future__ import annotations

import numpy as np

CM_PER_M = 100.0

# Foot-point noise as a fraction of box height, by confidence: (upper confidence, u, v).
# Robust SDs of RF-DETR Large against GT boxes on IMG_0106.
CONF_NOISE = ((0.5, 0.035, 0.065), (0.7, 0.030, 0.042), (0.8, 0.0225, 0.018), (0.9, 0.0136, 0.0102), (1.01, 0.009, 0.008))


class HomographyMapper:
    def __init__(self, image_to_court_cm: np.ndarray, u_ratio: float, v_ratio: float, conf_noise_scale: float = 0.0):
        self.InvA = np.diag([1 / CM_PER_M, 1 / CM_PER_M, 1.0]) @ image_to_court_cm
        self.u_ratio, self.v_ratio = u_ratio, v_ratio
        self.conf_noise_scale = conf_noise_scale  # above 0: use CONF_NOISE times this instead of the fixed ratios

    def uv2xy(self, uv: np.ndarray, sigma_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same projection and first-order covariance as upstream Mapper.uv2xy."""
        b = self.InvA @ np.array([[uv[0, 0]], [uv[1, 0]], [1.0]])
        gamma = 1 / b[2, 0]
        C = gamma * self.InvA[:2, :2] - (gamma**2) * b[:2, :] * self.InvA[2, :2]
        return b[:2, :] * gamma, C @ sigma_uv @ C.T

    def mapto(self, box: np.ndarray, conf: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """box: x, y, w, h in pixels. Noise is a fraction of box height, so it means the same at any resolution."""
        uv = np.array([[box[0] + box[2] / 2], [box[1] + box[3]]])
        u, v = self.u_ratio, self.v_ratio
        if self.conf_noise_scale:
            _, u, v = next(row for row in CONF_NOISE if conf <= row[0])
            u, v = u * self.conf_noise_scale, v * self.conf_noise_scale
        return self.uv2xy(uv, np.diag([(u * box[3]) ** 2, (v * box[3]) ** 2]))
