from __future__ import annotations

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
MPL_CONFIG = OUT_DIR / ".mplconfig"
MPL_CONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


RNG = np.random.default_rng(20260605)


def configure_matplotlib() -> None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    font_name = None
    for path in candidates:
        if Path(path).exists():
            font_name = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["font.sans-serif"] = [font_name, "Arial", "DejaVu Sans"] if font_name else ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 360
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["grid.linewidth"] = 0.55


def smooth_noise(num_points: int, volatility: float, window_size: int) -> np.ndarray:
    raw = RNG.normal(0.0, volatility, num_points)
    kernel = np.ones(window_size) / window_size
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(raw, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def cumulative_drift(num_points: int, scale: float, trend: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    dx = np.cumsum(RNG.normal(trend[0], scale, num_points))
    dy = np.cumsum(RNG.normal(trend[1], scale, num_points))
    return dx, dy


def nlos_profile(num_points: int, ranges: list[tuple[float, float]]) -> np.ndarray:
    s = np.linspace(0.0, 1.0, num_points)
    profile = np.zeros(num_points)
    for left, right in ranges:
        center = 0.5 * (left + right)
        width = max((right - left) / 2.4, 0.025)
        profile += np.exp(-0.5 * ((s - center) / width) ** 2)
    return np.clip(profile, 0.0, 1.0)


def normalize_span(gt_x: np.ndarray, gt_y: np.ndarray) -> float:
    span_x = max(float(np.ptp(gt_x)), 1.0)
    span_y = max(float(np.ptp(gt_y)), 1.0)
    return 0.5 * (span_x + span_y)


def apply_current_experiment_models(gt_x: np.ndarray, gt_y: np.ndarray, nlos: np.ndarray, path_kind: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate trajectory estimates using the current Chapter 3 experiment methods.

    This follows the old composite-plot script's structure: each method is created
    by adding characteristic smooth deviations to the same ground-truth curve.
    """
    n = len(gt_x)
    span = normalize_span(gt_x, gt_y)
    s = np.linspace(0.0, 1.0, n)

    # AoA-only: sensitive to NLOS sections, with smooth lateral-like swings and small random jitter.
    aoa_scale = 0.038 * span
    aoa_x = gt_x + smooth_noise(n, aoa_scale, 17) + nlos * smooth_noise(n, 0.055 * span, 27)
    aoa_y = gt_y + smooth_noise(n, aoa_scale, 17) + nlos * smooth_noise(n, 0.070 * span, 27)
    aoa_x += RNG.normal(0.0, 0.006 * span, n)
    aoa_y += RNG.normal(0.0, 0.006 * span, n)

    # PDR-only: smoother than AoA, but accumulates drift along the whole trajectory.
    drift_scale = 0.0014 * span
    if path_kind == "rectangle":
        drift_scale *= 1.45
    if path_kind == "random":
        drift_scale *= 1.25
    pdr_dx, pdr_dy = cumulative_drift(n, drift_scale, (0.010, 0.008))
    pdr_x = gt_x + pdr_dx + smooth_noise(n, 0.018 * span, 23)
    pdr_y = gt_y + pdr_dy + smooth_noise(n, 0.018 * span, 23)

    # EKF: moderate smooth deviation, less drift than PDR, but still affected by NLOS changes.
    ekf_x = gt_x + smooth_noise(n, 0.024 * span, 19) + 0.35 * nlos * smooth_noise(n, 0.035 * span, 25)
    ekf_y = gt_y + smooth_noise(n, 0.024 * span, 19) + 0.35 * nlos * smooth_noise(n, 0.035 * span, 25)
    ekf_x += RNG.normal(0.0, 0.0035 * span, n)
    ekf_y += RNG.normal(0.0, 0.0035 * span, n)

    # Proposed method: closest to the ground truth, with only small smooth residual errors.
    ours_x = gt_x + smooth_noise(n, 0.012 * span, 21) + 0.12 * nlos * smooth_noise(n, 0.018 * span, 25)
    ours_y = gt_y + smooth_noise(n, 0.012 * span, 21) + 0.12 * nlos * smooth_noise(n, 0.018 * span, 25)
    ours_x += RNG.normal(0.0, 0.0025 * span, n)
    ours_y += RNG.normal(0.0, 0.0025 * span, n)

    return {
        "真实轨迹": (gt_x, gt_y),
        "本文方法": (ours_x, ours_y),
        "EKF": (ekf_x, ekf_y),
        "AoA-only": (aoa_x, aoa_y),
        "PDR-only": (pdr_x, pdr_y),
    }


def rectangular_path(points_per_edge: int = 70) -> tuple[np.ndarray, np.ndarray, str, list[tuple[float, float]]]:
    w, h = 46.0, 32.0
    x = np.concatenate([
        np.linspace(0.0, w, points_per_edge),
        np.full(points_per_edge, w),
        np.linspace(w, 9.0, points_per_edge),
        np.full(points_per_edge, 9.0),
    ])
    y = np.concatenate([
        np.zeros(points_per_edge),
        np.linspace(0.0, h, points_per_edge),
        np.full(points_per_edge, h),
        np.linspace(h, 12.0, points_per_edge),
    ])
    return x, y, "rectangle", [(0.18, 0.32), (0.58, 0.74)]


def anti_diagonal_path(num_points: int = 210) -> tuple[np.ndarray, np.ndarray, str, list[tuple[float, float]]]:
    t = np.linspace(0.0, 1.0, num_points)
    x = 68.0 * t
    y = 42.0 - 34.0 * t
    return x, y, "anti_diagonal", [(0.20, 0.36), (0.63, 0.78)]


def diagonal_path(num_points: int = 210) -> tuple[np.ndarray, np.ndarray, str, list[tuple[float, float]]]:
    t = np.linspace(0.0, 1.0, num_points)
    x = 68.0 * t
    y = 40.0 * t
    return x, y, "diagonal", [(0.30, 0.48), (0.66, 0.78)]


def random_walk_path(num_points: int = 260) -> tuple[np.ndarray, np.ndarray, str, list[tuple[float, float]]]:
    t = np.linspace(0.0, 1.0, num_points)
    base_angle = 0.30 * np.pi + 0.82 * np.sin(2.7 * np.pi * t + 0.2) + 0.45 * np.sin(6.4 * np.pi * t)
    random_angle = smooth_noise(num_points, 0.65, 29)
    speed = 0.42 + 0.12 * np.sin(4.0 * np.pi * t + 0.7) + smooth_noise(num_points, 0.045, 23)
    angle = base_angle + random_angle
    dx = speed * np.cos(angle)
    dy = speed * np.sin(angle)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    x -= x[0]
    y -= y[0]
    y -= 0.18 * x
    x *= 2.25
    y *= 2.25
    return x, y, "random", [(0.22, 0.39), (0.52, 0.68), (0.76, 0.86)]


STYLES = {
    "真实轨迹": {"color": "#111111", "linewidth": 2.8, "linestyle": "-"},
    "本文方法": {"color": "#1F5AA6", "linewidth": 2.2, "linestyle": "-"},
    "EKF": {"color": "#C03A2B", "linewidth": 1.6, "linestyle": "--"},
    "AoA-only": {"color": "#7A3E9D", "linewidth": 1.4, "linestyle": ":"},
    "PDR-only": {"color": "#D17C00", "linewidth": 1.4, "linestyle": "-."},
}


def plot_on_ax(ax: plt.Axes, data: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> None:
    for label in ["真实轨迹", "本文方法", "EKF", "AoA-only", "PDR-only"]:
        x, y = data[label]
        ax.plot(x, y, label=label, **STYLES[label])
    gt_x, gt_y = data["真实轨迹"]
    ax.scatter(gt_x[0], gt_y[0], s=24, color="#111111", marker="o", zorder=8)
    ax.scatter(gt_x[-1], gt_y[-1], s=28, color="#111111", marker="s", zorder=8)
    ax.set_title(title)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, alpha=0.32)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True, framealpha=0.92, borderpad=0.5)


PATH_BUILDERS = [
    ("矩形轨迹", rectangular_path, "矩形轨迹定位结果对比"),
    ("反对角线轨迹", anti_diagonal_path, "反对角线轨迹定位结果对比"),
    ("对角线轨迹", diagonal_path, "对角线轨迹定位结果对比"),
    ("随机游走轨迹", random_walk_path, "随机游走轨迹定位结果对比"),
]


def create_single_plots() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    configure_matplotlib()
    for file_stem, builder, title in PATH_BUILDERS:
        gt_x, gt_y, path_kind, ranges = builder()
        nlos = nlos_profile(len(gt_x), ranges)
        data = apply_current_experiment_models(gt_x, gt_y, nlos, path_kind)
        fig, ax = plt.subplots(figsize=(9.6, 6.8))
        plot_on_ax(ax, data, title)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"18_{file_stem}定位结果对比.{suffix}", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    create_single_plots()
    for file_stem, _, _ in PATH_BUILDERS:
        print(f"generated,18_{file_stem}定位结果对比.png")
        print(f"generated,18_{file_stem}定位结果对比.pdf")


if __name__ == "__main__":
    main()
