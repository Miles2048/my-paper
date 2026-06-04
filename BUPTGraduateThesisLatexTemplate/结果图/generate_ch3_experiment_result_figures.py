from __future__ import annotations

import os
from dataclasses import dataclass
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


RNG = np.random.default_rng(20260604)
Q0 = np.full(3, 1.0 / 3.0)


@dataclass
class TrajectoryRecord:
    name: str
    rep: int
    truth: np.ndarray
    aoa: np.ndarray
    tr: np.ndarray
    rssi: np.ndarray
    equal: np.ndarray
    fixed: np.ndarray
    ekf: np.ndarray
    proposed: np.ndarray
    positions: np.ndarray
    q_star: np.ndarray
    q_hat: np.ndarray
    features: np.ndarray
    d12: np.ndarray
    d13: np.ndarray
    d23: np.ndarray
    nlos: np.ndarray
    idx: np.ndarray


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
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 320
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["xtick.labelsize"] = 8.7
    plt.rcParams["ytick.labelsize"] = 8.7
    plt.rcParams["legend.fontsize"] = 8.2
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["grid.linewidth"] = 0.55
    plt.rcParams["legend.frameon"] = False


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def resample_polyline(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    seg = points[1:] - points[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    n = int(round(cumulative[-1] / spacing)) + 1
    target = np.linspace(0.0, cumulative[-1], n)
    out = np.zeros((n, 2))
    for i, s in enumerate(target):
        idx = min(np.searchsorted(cumulative, s, side="right") - 1, len(lengths) - 1)
        local = (s - cumulative[idx]) / max(lengths[idx], 1e-9)
        out[i] = points[idx] + local * seg[idx]
    return out


def smooth_noise(n: int, scale: float) -> np.ndarray:
    raw = RNG.normal(0.0, scale, size=(n, 2))
    kernel = np.array([1, 2, 4, 5, 4, 2, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(raw, ((len(kernel) // 2, len(kernel) // 2), (0, 0)), mode="edge")
    return np.column_stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ]
    )


def smooth_1d(x: np.ndarray, window: int = 9) -> np.ndarray:
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    pad = window // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def trajectory_specs() -> list[tuple[str, np.ndarray]]:
    return [
        ("矩形轨迹", np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 14.0], [0.0, 14.0], [0.0, 8.0]])),
        ("对角线轨迹", np.array([[0.0, 6.0], [52.0, 36.0]])),
        ("反对角线轨迹", np.array([[0.0, 36.0], [52.0, 6.0]])),
        ("随机游走轨迹", np.array([[3.0, 5.0], [13.0, 12.0], [22.0, 9.0], [33.0, 20.0], [42.0, 16.0], [52.0, 30.0]])),
    ]


def nlos_mask(n: int, traj_idx: int) -> np.ndarray:
    ranges = [
        [(0.24, 0.40), (0.68, 0.80)],
        [(0.34, 0.51)],
        [(0.18, 0.34), (0.63, 0.74)],
        [(0.28, 0.43), (0.58, 0.71)],
    ][traj_idx]
    s = np.linspace(0.0, 1.0, n)
    mask = np.zeros(n, dtype=bool)
    for left, right in ranges:
        mask |= (s >= left) & (s <= right)
    return mask


def turn_mask(truth: np.ndarray, waypoints: np.ndarray) -> np.ndarray:
    if len(waypoints) <= 2:
        return np.zeros(len(truth), dtype=bool)
    mask = np.zeros(len(truth), dtype=bool)
    for point in waypoints[1:-1]:
        mask |= np.linalg.norm(truth - point, axis=1) < 4.0
    return mask


def temporal_filter(x: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    out = x.copy()
    for k in range(1, len(x)):
        out[k] = alpha * x[k] + (1.0 - alpha) * out[k - 1]
    return out


def solve_all_weights(positions: np.ndarray, truth: np.ndarray, mu: float) -> np.ndarray:
    weights = np.zeros((positions.shape[0], 3))
    for k in range(positions.shape[0]):
        pmat = positions[k]
        a = pmat.T @ pmat + mu * np.eye(3)
        b = pmat.T @ truth[k] + mu * Q0
        kkt = np.zeros((4, 4))
        kkt[:3, :3] = 2.0 * a
        kkt[:3, 3] = 1.0
        kkt[3, :3] = 1.0
        rhs = np.r_[2.0 * b, 1.0]
        weights[k] = np.linalg.solve(kkt, rhs)[:3]
    return weights


def enforce_affine_sum(weights: np.ndarray) -> np.ndarray:
    return weights - (weights.sum(axis=1, keepdims=True) - 1.0) / 3.0


def fuse(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ncs,ns->nc", positions, weights)


def simulate_record(name: str, waypoints: np.ndarray, traj_idx: int, rep: int) -> TrajectoryRecord:
    truth = resample_polyline(waypoints, spacing=1.0)
    n = len(truth)
    idx = np.arange(n)
    s = np.linspace(0.0, 1.0, n)
    nlos = nlos_mask(n, traj_idx)
    turns = turn_mask(truth, waypoints)

    phase = 0.6 * rep + 0.4 * traj_idx
    aoa_bias_dir = np.array([np.cos(1.4 + traj_idx), np.sin(1.4 + traj_idx)])
    rssi_bias_dir = np.array([np.cos(3.2 - 0.5 * traj_idx), np.sin(3.2 - 0.5 * traj_idx)])

    aoa = truth + smooth_noise(n, 0.95) + RNG.normal(0.0, 0.90, size=(n, 2))
    aoa += turns[:, None] * (smooth_noise(n, 1.0) + np.array([1.0, -0.8]))
    aoa[nlos] += 4.2 * aoa_bias_dir + RNG.normal(0.0, 1.25, size=(int(nlos.sum()), 2))

    rssi = truth + smooth_noise(n, 1.55) + RNG.normal(0.0, 1.65, size=(n, 2))
    rssi[nlos] += 6.4 * rssi_bias_dir + RNG.normal(0.0, 1.95, size=(int(nlos.sum()), 2))
    outlier_idx = RNG.choice(n, size=max(2, n // 26), replace=False)
    rssi[outlier_idx] += RNG.normal(0.0, 4.0, size=(len(outlier_idx), 2))

    drift_direction = np.array([np.cos(0.35 + 0.7 * traj_idx), np.sin(0.35 + 0.7 * traj_idx)])
    drift = (4.0 + 1.0 * traj_idx) * s[:, None] * drift_direction
    lagged_truth = truth.copy()
    lagged_truth[2:] = truth[:-2]
    turn_lag = lagged_truth - truth
    tr = truth + drift + 0.55 * turn_lag + smooth_noise(n, 0.75)
    tr += RNG.normal(0.0, 0.35, size=(n, 2))

    positions = np.stack([aoa, tr, rssi], axis=2)
    q_star = solve_all_weights(positions, truth, mu=80.0)

    d12 = np.linalg.norm(aoa - tr, axis=1)
    d13 = np.linalg.norm(aoa - rssi, axis=1)
    d23 = np.linalg.norm(tr - rssi, axis=1)
    rssi_error = np.linalg.norm(rssi - truth, axis=1)
    sigma_tau = 0.12 + 0.040 * np.sin(2.0 * np.pi * s + phase) ** 2 + 0.055 * nlos + RNG.normal(0.0, 0.010, n)
    rssi_var = smooth_1d(0.60 + 0.22 * rssi_error + 1.20 * nlos.astype(float) + RNG.normal(0.0, 0.18, n), 9)
    rbar = -55.0 - 0.10 * np.linalg.norm(truth - truth[0], axis=1) - 3.2 * nlos.astype(float) + RNG.normal(0.0, 0.9, n)
    features = np.column_stack([sigma_tau, rssi_var, rbar, d12, d13, d23])

    hetero = 0.024 + 0.004 * np.clip((d12 + d13 + d23) / 15.0, 0, 2)
    q_hat = q_star + RNG.normal(0.0, hetero[:, None], size=q_star.shape)
    q_hat = 0.92 * q_hat + 0.08 * Q0
    q_hat = enforce_affine_sum(q_hat)
    for k in range(1, n):
        q_hat[k] = 0.46 * q_hat[k - 1] + 0.54 * q_hat[k]
    q_hat = enforce_affine_sum(q_hat)

    equal = positions.mean(axis=2)
    fixed_weight = np.array([0.38, 0.34, 0.28])
    fixed = fuse(positions, np.tile(fixed_weight, (n, 1)))
    ekf_input = 0.82 * fixed + 0.18 * tr
    ekf = temporal_filter(ekf_input, alpha=0.62)
    proposed = fuse(positions, q_hat)

    return TrajectoryRecord(
        name=name,
        rep=rep,
        truth=truth,
        aoa=aoa,
        tr=tr,
        rssi=rssi,
        equal=equal,
        fixed=fixed,
        ekf=ekf,
        proposed=proposed,
        positions=positions,
        q_star=q_star,
        q_hat=q_hat,
        features=features,
        d12=d12,
        d13=d13,
        d23=d23,
        nlos=nlos,
        idx=idx,
    )


def generate_dataset(reps: int = 5) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    records = []
    first_rep = []
    for traj_idx, (name, waypoints) in enumerate(trajectory_specs()):
        for rep in range(reps):
            rec = simulate_record(name, waypoints, traj_idx, rep)
            records.append(rec)
            if rep == 0:
                first_rep.append(rec)
    return records, first_rep


def concat_records(records: list[TrajectoryRecord]) -> dict[str, np.ndarray]:
    keys = ["truth", "aoa", "tr", "rssi", "equal", "fixed", "ekf", "proposed", "q_star", "q_hat", "features", "d12", "d13", "d23"]
    out = {key: np.concatenate([getattr(rec, key) for rec in records], axis=0) for key in keys}
    out["traj_name"] = np.concatenate([[rec.name] * len(rec.truth) for rec in records])
    return out


def err(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(est - truth, axis=1)


def metrics(est: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    e = err(est, truth)
    return float(np.sqrt(np.mean(e**2))), float(np.mean(e)), float(np.percentile(e, 95)), float(np.max(e))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_nlos_path(ax: plt.Axes, rec: TrajectoryRecord) -> None:
    mask = rec.nlos
    if not np.any(mask):
        return
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value else i - 1
            ax.plot(rec.truth[start : end + 1, 0], rec.truth[start : end + 1, 1], color="#E9B872", lw=6.0, alpha=0.35, solid_capstyle="round")
            start = None


def add_nlos_bands(ax: plt.Axes, rec: TrajectoryRecord) -> None:
    mask = rec.nlos
    if not np.any(mask):
        return
    idx = rec.idx
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value else i - 1
            ax.axvspan(idx[start], idx[end], color="#EFE1C9", alpha=0.58, lw=0)
            start = None


def plot_typical_trajectories(records: list[TrajectoryRecord]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    axes = axes.ravel()
    style = [
        ("真实轨迹", "truth", "#111111", 2.2, "-"),
        ("AoA-only", "aoa", "#2F66B0", 1.05, "--"),
        ("PDR-only", "tr", "#7A5BA6", 1.05, "--"),
        ("RSSI-only", "rssi", "#C75050", 1.00, "--"),
        ("固定权重融合", "fixed", "#D9902F", 1.25, "-."),
        ("本文方法", "proposed", "#0E8F68", 2.05, "-"),
    ]
    for ax, rec in zip(axes, records):
        draw_nlos_path(ax, rec)
        for label, key, color, lw, ls in style:
            xy = getattr(rec, key)
            ax.plot(xy[:, 0], xy[:, 1], label=label, color=color, lw=lw, ls=ls)
        ax.scatter(rec.truth[0, 0], rec.truth[0, 1], marker="o", s=32, color="#111111", zorder=5)
        ax.scatter(rec.truth[-1, 0], rec.truth[-1, 1], marker="s", s=34, color="#111111", zorder=5)
        ax.set_title(rec.name)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.grid(True, alpha=0.24)
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#E9B872", lw=6, alpha=0.45))
    labels.append("NLOS 区段")
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), columnspacing=1.8)
    save(fig, "01_轨迹对比示意")


def plot_cdf(data: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    methods = [
        ("AoA-only", "aoa", "#2F66B0"),
        ("PDR-only", "tr", "#7A5BA6"),
        ("RSSI-only", "rssi", "#C75050"),
        ("等权融合", "equal", "#8A8A8A"),
        ("固定权重融合", "fixed", "#D9902F"),
        ("EKF", "ekf", "#4B8F9E"),
        ("本文方法", "proposed", "#0E8F68"),
    ]
    for label, key, color in methods:
        e = np.sort(err(data[key], data["truth"]))
        y = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, y, lw=2.25 if label == "本文方法" else 1.35, color=color, label=label)
    ax.axhline(0.95, color="#666666", lw=0.8, ls=":", alpha=0.75)
    ax.set_xlabel("定位误差 / m")
    ax.set_ylabel("累计概率")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", ncol=2)
    save(fig, "02_定位误差CDF示意")


def plot_metric_bar(data: dict[str, np.ndarray]) -> None:
    labels = ["AoA", "PDR", "RSSI", "等权", "固定权重", "EKF", "本文方法"]
    keys = ["aoa", "tr", "rssi", "equal", "fixed", "ekf", "proposed"]
    values = np.array([metrics(data[key], data["truth"])[:3] for key in keys])
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    x = np.arange(len(labels))
    width = 0.23
    colors = ["#4F78BD", "#7FA35C", "#C95B5B"]
    metric_labels = ["RMSE", "MAE", "P95"]
    for j in range(3):
        bars = ax.bar(x + (j - 1) * width, values[:, j], width=width, label=metric_labels[j], color=colors[j])
        for bar in bars:
            if bar.get_height() < 4.2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.0, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("定位误差 / m")
    ax.grid(True, axis="y", alpha=0.26)
    ax.legend(ncol=3, loc="upper left")
    save(fig, "03_误差指标柱状图示意")


def plot_weight_sequence(rec: TrajectoryRecord) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    add_nlos_bands(ax, rec)
    labels = [r"$\hat q_{\mathrm{AoA}}$", r"$\hat q_{\mathrm{PDR}}$", r"$\hat q_{\mathrm{RSSI}}$"]
    colors = ["#2F66B0", "#7A5BA6", "#C75050"]
    for i in range(3):
        ax.plot(rec.idx, rec.q_hat[:, i], color=colors[i], lw=1.85, label=labels[i])
    ax.axhline(1.0 / 3.0, color="#707070", lw=0.85, ls=":", alpha=0.85)
    ymin = float(rec.q_hat.min()) - 0.035
    ymax = float(rec.q_hat.max()) + 0.035
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("样本序号 k")
    ax.set_ylabel("XGBoost 预测融合权重")
    ax.grid(True, alpha=0.24)
    ax.legend(ncol=3, loc="upper right")
    ylim = ax.get_ylim()
    ax.text(rec.idx[int(len(rec.idx) * 0.29)], ylim[1] - 0.08 * (ylim[1] - ylim[0]), "NLOS/遮挡区", color="#8B6B27", fontsize=8.8)
    save(fig, "04_预测权重时序示意")


def plot_weight_scatter(data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    panels = [
        (0, "AoA 权重", "#2F66B0"),
        (1, "PDR 权重", "#7A5BA6"),
        (2, "RSSI 权重", "#C75050"),
    ]
    for ax, (i, title, color) in zip(axes, panels):
        x = data["q_star"][:, i]
        y = data["q_hat"][:, i]
        lo = min(float(x.min()), float(y.min())) - 0.06
        hi = max(float(x.max()), float(y.max())) + 0.06
        ax.scatter(x, y, s=12, alpha=0.44, color=color, edgecolor="white", linewidth=0.15)
        ax.plot([lo, hi], [lo, hi], color="#222222", lw=0.95, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("离线反解权重标签")
        ax.set_ylabel("XGBoost预测权重")
        ax.grid(True, alpha=0.23)
    save(fig, "05_反解权重与预测权重散点示意")


def plot_consistency_relationship(rec: TrajectoryRecord) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.1), sharex=True)
    add_nlos_bands(axes[0], rec)
    axes[0].plot(rec.idx, rec.d12, color="#2F66B0", lw=1.45, label=r"$d_{12}$: AoA-PDR")
    axes[0].plot(rec.idx, rec.d13, color="#D9902F", lw=1.45, label=r"$d_{13}$: AoA-RSSI")
    axes[0].plot(rec.idx, rec.d23, color="#C75050", lw=1.45, label=r"$d_{23}$: PDR-RSSI")
    axes[0].set_ylabel("一致性距离 / m")
    axes[0].grid(True, alpha=0.24)
    axes[0].legend(ncol=3, loc="upper left")

    add_nlos_bands(axes[1], rec)
    axes[1].plot(rec.idx, rec.q_hat[:, 0], color="#2F66B0", lw=1.65, label=r"$\hat q_{\mathrm{AoA}}$")
    axes[1].plot(rec.idx, rec.q_hat[:, 1], color="#7A5BA6", lw=1.65, label=r"$\hat q_{\mathrm{PDR}}$")
    axes[1].plot(rec.idx, rec.q_hat[:, 2], color="#C75050", lw=1.65, label=r"$\hat q_{\mathrm{RSSI}}$")
    axes[1].axhline(0.0, color="#555555", lw=0.75, alpha=0.50)
    axes[1].set_xlabel("样本序号 k")
    axes[1].set_ylabel("预测融合权重")
    axes[1].grid(True, alpha=0.24)
    axes[1].legend(ncol=3, loc="upper right")
    save(fig, "06_一致性距离与权重关系示意")


def plot_ablation_and_feature_importance(data: dict[str, np.ndarray]) -> None:
    base_rmse, _, base_p95, _ = metrics(data["proposed"], data["truth"])
    equal_rmse, _, equal_p95, _ = metrics(data["equal"], data["truth"])
    fixed_rmse, _, fixed_p95, _ = metrics(data["fixed"], data["truth"])
    variants = ["完整方法", "去除一致性距离", "去除物理特征", "无正则反解", "无仿射归一化", "等权融合"]
    rmse = np.array([base_rmse, base_rmse * 1.17, base_rmse * 1.27, base_rmse * 1.33, fixed_rmse * 1.07, equal_rmse])
    p95 = np.array([base_p95, base_p95 * 1.21, base_p95 * 1.31, base_p95 * 1.43, fixed_p95 * 1.07, equal_p95])
    features = [r"$\sigma_{\tau}$", r"$\sigma^2_{\mathrm{RSSI}}$", r"$\bar r$", r"$d_{12}$", r"$d_{13}$", r"$d_{23}$"]
    importance = np.array([0.145, 0.181, 0.083, 0.154, 0.226, 0.211])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), gridspec_kw={"width_ratios": [1.45, 1.0]})
    x = np.arange(len(variants))
    width = 0.34
    axes[0].bar(x - width / 2, rmse, width=width, color="#4F78BD", label="RMSE")
    axes[0].bar(x + width / 2, p95, width=width, color="#C95B5B", label="P95")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(variants, rotation=20, ha="right")
    axes[0].set_ylabel("定位误差 / m")
    axes[0].set_title("消融实验")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(ncol=2, loc="upper left")

    order = np.argsort(importance)
    axes[1].barh(np.array(features)[order], importance[order], color=["#8E83B9", "#8E83B9", "#5E9B8A", "#5E9B8A", "#5E9B8A", "#5E9B8A"])
    axes[1].set_xlabel("平均特征重要性")
    axes[1].set_title("XGBoost 特征重要性")
    axes[1].grid(True, axis="x", alpha=0.25)
    for y, v in enumerate(importance[order]):
        axes[1].text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8.0)
    axes[1].set_xlim(0, 0.27)
    save(fig, "07_消融实验柱状图示意")


def plot_snr_robustness() -> None:
    snr = np.array([-5, 0, 5, 10, 15, 20])
    rmse_curves = {
        "AoA-only": ("#2F66B0", np.array([8.20, 6.55, 4.82, 3.72, 3.25, 3.04])),
        "PDR-only": ("#7A5BA6", np.array([4.92, 4.72, 4.55, 4.43, 4.36, 4.32])),
        "RSSI-only": ("#C75050", np.array([9.85, 7.42, 5.74, 4.78, 4.32, 4.03])),
        "等权融合": ("#8A8A8A", np.array([6.22, 5.06, 4.03, 3.28, 2.96, 2.78])),
        "固定权重融合": ("#D9902F", np.array([5.72, 4.62, 3.63, 2.96, 2.63, 2.45])),
        "本文方法": ("#0E8F68", np.array([4.58, 3.64, 2.88, 2.32, 2.04, 1.92])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), sharex=True)
    for label, (color, values) in rmse_curves.items():
        axes[0].plot(snr, values, marker="o", ms=4.2, lw=2.2 if label == "本文方法" else 1.35, color=color, label=label)
    axes[0].set_xlabel("SNR / dB")
    axes[0].set_ylabel("RMSE / m")
    axes[0].set_title("平均误差随 SNR 变化")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend(loc="upper right", fontsize=7.6)

    proposed_p95 = np.array([7.65, 6.18, 4.82, 3.86, 3.38, 3.12])
    fixed_p95 = np.array([9.82, 7.92, 6.25, 5.10, 4.55, 4.23])
    equal_p95 = np.array([10.78, 8.78, 6.92, 5.78, 5.08, 4.82])
    axes[1].plot(snr, equal_p95, marker="o", color="#8A8A8A", lw=1.55, label="等权融合 P95")
    axes[1].plot(snr, fixed_p95, marker="s", color="#D9902F", lw=1.55, label="固定权重 P95")
    axes[1].plot(snr, proposed_p95, marker="o", color="#0E8F68", lw=2.25, label="本文方法 P95")
    axes[1].set_xlabel("SNR / dB")
    axes[1].set_ylabel("P95 / m")
    axes[1].set_title("尾部误差随 SNR 变化")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend(loc="upper right")
    save(fig, "08_不同SNR鲁棒性示意")


def plot_regularization_effect() -> None:
    labels = ["0", "10", "30", "80", "200", "500", "1000"]
    x = np.arange(len(labels))
    inverse_mae = np.array([1.54, 1.61, 1.69, 1.83, 2.04, 2.34, 2.68])
    weight_norm = np.array([5.72, 3.74, 2.46, 1.74, 1.25, 0.94, 0.76])
    extreme_ratio = np.array([18.2, 10.6, 5.7, 2.4, 1.2, 0.6, 0.4])

    fig, ax1 = plt.subplots(figsize=(7.6, 4.9))
    ax2 = ax1.twinx()
    ax1.plot(x, inverse_mae, color="#2F66B0", marker="o", lw=1.8, label="平均反解误差")
    ax2.plot(x, weight_norm, color="#C75050", marker="s", lw=1.8, label="平均权重范数")
    ax2.plot(x, extreme_ratio / 4.0, color="#D9902F", marker="^", lw=1.45, ls="--", label="极端权重比例/4")
    ax1.axvline(3, color="#0E8F68", lw=1.0, ls=":", alpha=0.9)
    ax1.text(3.08, inverse_mae.min() + 0.08, r"本文取值 $\mu=80$", color="#0E8F68", fontsize=8.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel(r"正则化系数 $\mu$")
    ax1.set_ylabel("平均反解误差 / m", color="#2F66B0")
    ax2.set_ylabel(r"权重幅度统计", color="#C75050")
    ax1.grid(True, alpha=0.28)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    save(fig, "09_正则化系数影响示意")


def plot_scene_performance(records: list[TrajectoryRecord]) -> None:
    names = [rec.name for rec in records]
    methods = [
        ("AoA", "aoa", "#2F66B0"),
        ("PDR", "tr", "#7A5BA6"),
        ("RSSI", "rssi", "#C75050"),
        ("本文方法", "proposed", "#0E8F68"),
    ]
    values = np.array([[metrics(getattr(rec, key), rec.truth)[0] for _, key, _ in methods] for rec in records])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(names))
    width = 0.18
    for j, (label, _, color) in enumerate(methods):
        ax.bar(x + (j - 1.5) * width, values[:, j], width=width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE / m")
    ax.set_title("不同轨迹类型下的定位误差")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper left")
    save(fig, "14_不同轨迹类型定位误差示意")


def plot_inverse_weight_distribution(data: dict[str, np.ndarray]) -> None:
    q = data["q_star"]
    e_star = err(fuse(np.stack([data["aoa"], data["tr"], data["rssi"]], axis=2), q), data["truth"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    parts = axes[0].violinplot([q[:, 0], q[:, 1], q[:, 2]], showmeans=True, showextrema=True, widths=0.78)
    for pc, color in zip(parts["bodies"], ["#2F66B0", "#7A5BA6", "#C75050"]):
        pc.set_facecolor(color)
        pc.set_edgecolor("#333333")
        pc.set_alpha(0.45)
    for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
        parts[key].set_color("#333333")
        parts[key].set_linewidth(0.9)
    axes[0].axhline(0, color="#555555", lw=0.8, alpha=0.55)
    axes[0].axhline(1 / 3, color="#777777", lw=0.8, ls=":", alpha=0.8)
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(["AoA", "PDR", "RSSI"])
    axes[0].set_ylabel("反解权重标签")
    axes[0].set_title("离线反解权重分布")
    axes[0].grid(True, axis="y", alpha=0.24)

    axes[1].hist(e_star, bins=36, color="#5E9B8A", alpha=0.72, edgecolor="white")
    axes[1].axvline(np.mean(e_star), color="#111111", lw=1.2, label=f"均值 {np.mean(e_star):.2f} m")
    axes[1].axvline(np.percentile(e_star, 95), color="#C75050", lw=1.2, ls="--", label=f"P95 {np.percentile(e_star, 95):.2f} m")
    axes[1].set_xlabel("反解融合误差 / m")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("反解误差分布")
    axes[1].grid(True, axis="y", alpha=0.24)
    axes[1].legend()
    save(fig, "15_离线反解权重标签质量示意")


def write_summary(data: dict[str, np.ndarray], records: list[TrajectoryRecord]) -> None:
    rows = [
        ("AoA-only", "aoa"),
        ("Trajectory-recursion-only", "tr"),
        ("RSSI-only", "rssi"),
        ("Equal fusion", "equal"),
        ("Fixed-weight fusion", "fixed"),
        ("EKF", "ekf"),
        ("Proposed XGBoost affine fusion", "proposed"),
    ]
    lines = [
        "# 第三章实验结果参考图说明",
        "",
        "本目录中的实验结果图由 `generate_ch3_experiment_result_figures.py` 使用 Python 生成。数据为按论文设定构造的可复现实验参考数据，用于确定图形形式、指标量级和预期趋势；正式论文应替换为真实实验统计值。",
        "",
        "## 总体定位指标",
        "",
        "| 方法 | RMSE (m) | MAE (m) | P95 (m) | MaxError (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in rows:
        rmse, mae, p95, max_error = metrics(data[key], data["truth"])
        lines.append(f"| {label} | {rmse:.3f} | {mae:.3f} | {p95:.3f} | {max_error:.3f} |")
    weight_mae = np.mean(np.abs(data["q_hat"] - data["q_star"]), axis=0)
    weight_rmse = np.sqrt(np.mean((data["q_hat"] - data["q_star"]) ** 2, axis=0))
    corr = [np.corrcoef(data["q_hat"][:, i], data["q_star"][:, i])[0, 1] for i in range(3)]
    lines.extend(
        [
            "",
            "## 权重预测参考指标",
            "",
            "| 权重分量 | MAE | RMSE | 相关系数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, i in [("AoA", 0), ("Trajectory recursion", 1), ("RSSI", 2)]:
        lines.append(f"| {label} | {weight_mae[i]:.3f} | {weight_rmse[i]:.3f} | {corr[i]:.3f} |")
    total_distance = sum(polyline_length(points) for _, points in trajectory_specs())
    lines.extend(
        [
            "",
            f"四类测试轨迹唯一物理里程约为 {total_distance:.1f} m，主实验 SNR 设为 10 dB，图中采用 5 次观测扰动重复以形成更稳定的统计曲线和权重散点。",
            "",
            "## 生成文件",
            "",
            "- `01_轨迹对比示意.png/pdf`：四类典型轨迹定位结果对比。",
            "- `02_定位误差CDF示意.png/pdf`：不同方法定位误差 CDF。",
            "- `03_误差指标柱状图示意.png/pdf`：RMSE、MAE、P95 对比。",
            "- `04_预测权重时序示意.png/pdf`：典型轨迹上的 XGBoost 预测权重。",
            "- `05_反解权重与预测权重散点示意.png/pdf`：测试集预测权重与反解标签对比。",
            "- `06_一致性距离与权重关系示意.png/pdf`：一致性距离和预测权重时序关系。",
            "- `07_消融实验柱状图示意.png/pdf`：消融实验和特征重要性。",
            "- `08_不同SNR鲁棒性示意.png/pdf`：不同 SNR 下 RMSE/P95 变化。",
            "- `09_正则化系数影响示意.png/pdf`：正则化系数对反解质量影响。",
            "- `14_不同轨迹类型定位误差示意.png/pdf`：分轨迹类型 RMSE 对比。",
            "- `15_离线反解权重标签质量示意.png/pdf`：反解权重标签和反解误差分布。",
        ]
    )
    (OUT_DIR / "README_第三章实验结果参考图.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    records, first_rep = generate_dataset(reps=5)
    data = concat_records(records)
    plot_typical_trajectories(first_rep)
    plot_cdf(data)
    plot_metric_bar(data)
    plot_weight_sequence(first_rep[0])
    plot_weight_scatter(data)
    plot_consistency_relationship(first_rep[0])
    plot_ablation_and_feature_importance(data)
    plot_snr_robustness()
    plot_regularization_effect()
    plot_scene_performance(first_rep)
    plot_inverse_weight_distribution(data)
    write_summary(data, first_rep)
    for label, key in [
        ("AoA-only", "aoa"),
        ("PDR-only", "tr"),
        ("RSSI-only", "rssi"),
        ("Equal", "equal"),
        ("Fixed", "fixed"),
        ("EKF", "ekf"),
        ("Proposed", "proposed"),
    ]:
        rmse, mae, p95, max_error = metrics(data[key], data["truth"])
        print(f"{label:>10s}: RMSE={rmse:.3f}, MAE={mae:.3f}, P95={p95:.3f}, Max={max_error:.3f}")


if __name__ == "__main__":
    main()
