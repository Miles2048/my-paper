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
from matplotlib.patches import Rectangle


RNG = np.random.default_rng(20260604)


def configure_matplotlib() -> None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    font_name = None
    for path in candidates:
        if Path(path).exists():
            font_name = fm.FontProperties(fname=path).get_name()
            break
    if font_name is not None:
        plt.rcParams["font.sans-serif"] = [font_name, "Arial", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 8.5
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["grid.linewidth"] = 0.55
    plt.rcParams["legend.frameon"] = False


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    seg = points[1:] - points[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    target = np.linspace(0.0, cumulative[-1], n)
    out = np.zeros((n, 2))
    for i, s in enumerate(target):
        idx = min(np.searchsorted(cumulative, s, side="right") - 1, len(lengths) - 1)
        local = (s - cumulative[idx]) / max(lengths[idx], 1e-9)
        out[i] = points[idx] + local * seg[idx]
    return out


def smooth_noise(n: int, scale: float) -> np.ndarray:
    raw = RNG.normal(0.0, scale, size=(n, 2))
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
    return np.column_stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ]
    )


def generate_synthetic_localization(n: int = 720) -> dict[str, np.ndarray]:
    waypoints = np.array(
        [
            [0.0, 0.0],
            [36.0, 0.0],
            [36.0, 28.0],
            [78.0, 28.0],
            [96.0, 8.0],
            [122.0, 42.0],
        ]
    )
    truth = resample_polyline(waypoints, n)

    idx = np.arange(n)
    nlos_1 = (idx >= 58) & (idx <= 106)
    nlos_2 = (idx >= 164) & (idx <= 214)
    turn_zone = ((idx >= 40) & (idx <= 75)) | ((idx >= 126) & (idx <= 170))

    aoa = truth + RNG.normal(0.0, 1.55, size=(n, 2))
    aoa[nlos_1] += np.array([5.8, -3.6]) + RNG.normal(0.0, 2.1, size=(nlos_1.sum(), 2))
    aoa[nlos_2] += np.array([-4.8, 4.4]) + RNG.normal(0.0, 1.8, size=(nlos_2.sum(), 2))
    aoa[turn_zone] += smooth_noise(n, 1.0)[turn_zone]

    rssi = truth + RNG.normal(0.0, 2.7, size=(n, 2))
    rssi[nlos_1] += np.array([8.5, 6.0]) + RNG.normal(0.0, 3.0, size=(nlos_1.sum(), 2))
    rssi[nlos_2] += np.array([-7.6, -5.0]) + RNG.normal(0.0, 3.3, size=(nlos_2.sum(), 2))

    drift = np.column_stack([np.linspace(0, 7.5, n), np.linspace(0, -5.8, n)])
    tr = truth + drift + smooth_noise(n, 1.25)
    tr[idx > 150] += np.column_stack(
        [
            np.linspace(0, 5.2, (idx > 150).sum()),
            np.linspace(0, -3.4, (idx > 150).sum()),
        ]
    )

    positions = np.stack([aoa, tr, rssi], axis=2)
    q_star = solve_all_weights(positions, truth, mu=80.0)
    q_hat = q_star + RNG.normal(0.0, 0.03, size=q_star.shape)
    q_hat = enforce_affine_sum(q_hat)
    for k in range(1, n):
        q_hat[k] = 0.55 * q_hat[k - 1] + 0.45 * q_hat[k]
    q_hat = enforce_affine_sum(q_hat)

    proposed = fuse(positions, q_hat)
    equal = positions.mean(axis=2)
    fixed_weight = np.array([0.36, 0.34, 0.30])
    fixed = fuse(positions, np.tile(fixed_weight, (n, 1)))

    d12 = np.linalg.norm(aoa - tr, axis=1)
    d13 = np.linalg.norm(aoa - rssi, axis=1)
    d23 = np.linalg.norm(tr - rssi, axis=1)

    return {
        "truth": truth,
        "aoa": aoa,
        "tr": tr,
        "rssi": rssi,
        "positions": positions,
        "q_star": q_star,
        "q_hat": q_hat,
        "proposed": proposed,
        "equal": equal,
        "fixed": fixed,
        "d12": d12,
        "d13": d13,
        "d23": d23,
        "nlos_1": nlos_1,
        "nlos_2": nlos_2,
        "idx": idx,
    }


def solve_all_weights(positions: np.ndarray, truth: np.ndarray, mu: float) -> np.ndarray:
    n = positions.shape[0]
    q0 = np.full(3, 1.0 / 3.0)
    weights = np.zeros((n, 3))
    for k in range(n):
        pmat = positions[k]
        a = pmat.T @ pmat + mu * np.eye(3)
        b = pmat.T @ truth[k] + mu * q0
        kkt = np.zeros((4, 4))
        kkt[:3, :3] = 2 * a
        kkt[:3, 3] = 1.0
        kkt[3, :3] = 1.0
        rhs = np.r_[2 * b, 1.0]
        weights[k] = np.linalg.solve(kkt, rhs)[:3]
    return weights


def enforce_affine_sum(weights: np.ndarray) -> np.ndarray:
    return weights - (weights.sum(axis=1, keepdims=True) - 1.0) / 3.0


def fuse(positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("ncs,ns->nc", positions, weights)


def err(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(est - truth, axis=1)


def metrics(est: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    e = err(est, truth)
    return float(np.sqrt(np.mean(e**2))), float(np.mean(e)), float(np.percentile(e, 95))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_nlos_bands(ax: plt.Axes, idx: np.ndarray, nlos_1: np.ndarray, nlos_2: np.ndarray) -> None:
    for mask in (nlos_1, nlos_2):
        left = idx[mask][0]
        right = idx[mask][-1]
        ax.axvspan(left, right, color="#EFE6D8", alpha=0.72, lw=0)


def plot_trajectory(data: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    series = [
        ("真实轨迹", data["truth"], "#111111", 2.5, "-"),
        ("AoA-only", data["aoa"], "#3B6FB6", 1.35, "--"),
        ("轨迹递推-only", data["tr"], "#7A5BA6", 1.35, "--"),
        ("RSSI-only", data["rssi"], "#C75C5C", 1.25, "--"),
        ("等权融合", data["equal"], "#8A8A8A", 1.35, "-."),
        ("本文方法", data["proposed"], "#0E9F6E", 2.3, "-"),
    ]
    for label, xy, color, lw, ls in series:
        ax.plot(xy[:, 0], xy[:, 1], label=label, color=color, lw=lw, ls=ls)
    ax.scatter(data["truth"][0, 0], data["truth"][0, 1], s=34, color="#111111", marker="o", zorder=5)
    ax.scatter(data["truth"][-1, 0], data["truth"][-1, 1], s=40, color="#111111", marker="s", zorder=5)
    ax.text(data["truth"][0, 0] + 1.8, data["truth"][0, 1] - 2.2, "起点", fontsize=8.5)
    ax.text(data["truth"][-1, 0] + 1.8, data["truth"][-1, 1] + 0.3, "终点", fontsize=8.5)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, alpha=0.28)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.5)
    save(fig, "01_轨迹对比示意")


def plot_cdf(data: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    methods = [
        ("AoA-only", data["aoa"], "#3B6FB6"),
        ("轨迹递推-only", data["tr"], "#7A5BA6"),
        ("RSSI-only", data["rssi"], "#C75C5C"),
        ("等权融合", data["equal"], "#8A8A8A"),
        ("固定权重融合", data["fixed"], "#D9902F"),
        ("本文方法", data["proposed"], "#0E9F6E"),
    ]
    for label, est, color in methods:
        e = np.sort(err(est, data["truth"]))
        y = np.linspace(0, 1, len(e), endpoint=False)
        ax.plot(e, y, lw=2.0 if label == "本文方法" else 1.35, color=color, label=label)
    ax.set_xlabel("定位误差 / m")
    ax.set_ylabel("累计概率")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, "02_定位误差CDF示意")


def plot_metric_bar(data: dict[str, np.ndarray]) -> None:
    labels = ["AoA", "轨迹递推", "RSSI", "等权", "固定权重", "本文方法"]
    estimates = [data["aoa"], data["tr"], data["rssi"], data["equal"], data["fixed"], data["proposed"]]
    values = np.array([metrics(est, data["truth"]) for est in estimates])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(labels))
    width = 0.24
    colors = ["#597DBF", "#81A969", "#C75C5C"]
    metric_labels = ["RMSE", "MAE", "P95"]
    for j in range(3):
        ax.bar(x + (j - 1) * width, values[:, j], width=width, label=metric_labels[j], color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("定位误差 / m")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(ncol=3, loc="upper right")
    save(fig, "03_误差指标柱状图示意")


def plot_weights(data: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    idx = data["idx"]
    add_nlos_bands(ax, idx, data["nlos_1"], data["nlos_2"])
    labels = [r"$\hat{q}_{AoA}$", r"$\hat{q}_{TR}$", r"$\hat{q}_{RSSI}$"]
    colors = ["#3B6FB6", "#7A5BA6", "#C75C5C"]
    for i in range(3):
        ax.plot(idx, data["q_hat"][:, i], color=colors[i], lw=1.8, label=labels[i])
    ax.axhline(0, color="#555555", lw=0.8, alpha=0.5)
    ax.axhline(1 / 3, color="#777777", lw=0.8, ls=":", alpha=0.8)
    ax.set_xlabel("样本序号 k")
    ax.set_ylabel("预测融合权重")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3)
    ax.text(62, ax.get_ylim()[1] * 0.90, "NLOS/遮挡区", fontsize=9, color="#8A6A2A")
    save(fig, "04_预测权重时序示意")


def plot_weight_scatter(data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharex=False, sharey=False)
    panels = [
        (0, "AoA 权重", "#2F66B0"),
        (2, "RSSI 权重", "#C75050"),
    ]
    for ax, (weight_idx, title, color) in zip(axes, panels):
        x = data["q_star"][:, weight_idx]
        y = data["q_hat"][:, weight_idx]
        lo = min(x.min(), y.min()) - 0.035
        hi = max(x.max(), y.max()) + 0.035

        ax.scatter(
            x,
            y,
            s=11,
            alpha=0.48,
            color=color,
            edgecolor="white",
            linewidth=0.18,
        )
        ax.plot([lo, hi], [lo, hi], color="#202020", lw=1.0, ls="--", label=r"$y=x$")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("反解权重标签")
        ax.set_ylabel("XGBoost预测权重")
        ax.grid(True, alpha=0.24)
        ax.legend(loc="lower right", fontsize=7.8)
    save(fig, "05_反解权重与预测权重散点示意")


def plot_consistency_relationship(data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0), sharex=True)
    idx = data["idx"]
    add_nlos_bands(axes[0], idx, data["nlos_1"], data["nlos_2"])
    axes[0].plot(idx, data["d12"], color="#3B6FB6", lw=1.5, label=r"$d_{AoA,TR}$")
    axes[0].plot(idx, data["d13"], color="#D9902F", lw=1.5, label=r"$d_{AoA,RSSI}$")
    axes[0].plot(idx, data["d23"], color="#C75C5C", lw=1.5, label=r"$d_{TR,RSSI}$")
    axes[0].set_ylabel("一致性距离 / m")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, loc="upper left")

    add_nlos_bands(axes[1], idx, data["nlos_1"], data["nlos_2"])
    axes[1].plot(idx, data["q_hat"][:, 0], color="#3B6FB6", lw=1.7, label=r"$\hat{q}_{AoA}$")
    axes[1].plot(idx, data["q_hat"][:, 1], color="#7A5BA6", lw=1.7, label=r"$\hat{q}_{TR}$")
    axes[1].plot(idx, data["q_hat"][:, 2], color="#C75C5C", lw=1.7, label=r"$\hat{q}_{RSSI}$")
    axes[1].set_xlabel("样本序号 k")
    axes[1].set_ylabel("预测融合权重")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, loc="upper right")
    save(fig, "06_一致性距离与权重关系示意")


def plot_ablation(data: dict[str, np.ndarray]) -> None:
    base_rmse, _, base_p95 = metrics(data["proposed"], data["truth"])
    values = np.array(
        [
            [base_rmse, base_p95],
            [base_rmse * 1.18, base_p95 * 1.22],
            [base_rmse * 1.31, base_p95 * 1.36],
            [base_rmse * 1.42, base_p95 * 1.49],
            [metrics(data["equal"], data["truth"])[0], metrics(data["equal"], data["truth"])[2]],
            [metrics(data["fixed"], data["truth"])[0], metrics(data["fixed"], data["truth"])[2]],
        ]
    )
    labels = ["完整方法", "无一致性距离", "无物理特征", "无正则反解", "等权融合", "固定权重"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width / 2, values[:, 0], width=width, color="#597DBF", label="RMSE")
    ax.bar(x + width / 2, values[:, 1], width=width, color="#C75C5C", label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("定位误差 / m")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(ncol=2, loc="upper left")
    save(fig, "07_消融实验柱状图示意")


def plot_snr_robustness(data: dict[str, np.ndarray]) -> None:
    snr = np.array([-5, 0, 5, 10, 15, 20])
    methods = {
        "AoA-only": ("#3B6FB6", np.array([10.8, 8.3, 6.1, 4.9, 4.2, 3.8])),
        "RSSI-only": ("#C75C5C", np.array([13.2, 10.0, 7.6, 5.9, 4.9, 4.4])),
        "等权融合": ("#8A8A8A", np.array([9.3, 7.1, 5.4, 4.4, 3.8, 3.5])),
        "固定权重融合": ("#D9902F", np.array([8.8, 6.6, 5.0, 4.1, 3.5, 3.2])),
        "本文方法": ("#0E9F6E", np.array([6.3, 5.1, 4.0, 3.2, 2.8, 2.6])),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, (color, rmse) in methods.items():
        ax.plot(snr, rmse, marker="o", lw=2.1 if label == "本文方法" else 1.45, color=color, label=label)
    ax.set_xlabel("SNR / dB")
    ax.set_ylabel("RMSE / m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, "08_不同SNR鲁棒性示意")


def plot_regularization_effect(data: dict[str, np.ndarray]) -> None:
    mus = np.array([0, 20, 80, 200, 520, 1000, 1800], dtype=float)
    inv_error = np.array([1.85, 1.92, 2.05, 2.22, 2.42, 2.73, 3.18])
    weight_norm = np.array([5.8, 3.4, 2.05, 1.35, 0.92, 0.74, 0.62])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax2 = ax1.twinx()
    ax1.plot(mus, inv_error, color="#3B6FB6", marker="o", lw=1.8, label="反解融合误差")
    ax2.plot(mus, weight_norm, color="#C75C5C", marker="s", lw=1.8, label="权重范数")
    ax1.set_xlabel(r"正则化系数 $\mu$")
    ax1.set_ylabel("反解融合误差 / m", color="#3B6FB6")
    ax2.set_ylabel(r"平均权重范数 $\|\mathbf{q}^\ast\|_2$", color="#C75C5C")
    ax1.grid(True, alpha=0.28)
    ax1.set_xscale("symlog", linthresh=20)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    save(fig, "09_正则化系数影响示意")


def write_summary(data: dict[str, np.ndarray]) -> None:
    rows = [
        ("AoA-only", data["aoa"]),
        ("Trajectory-recursion-only", data["tr"]),
        ("RSSI-only", data["rssi"]),
        ("Equal fusion", data["equal"]),
        ("Fixed-weight fusion", data["fixed"]),
        ("Proposed supervised affine fusion", data["proposed"]),
    ]
    lines = [
        "# Expected result figure schematic summary",
        "",
        "These figures are generated from synthetic data only. They show the expected qualitative behavior of the proposed method and must not be reported as real experimental results.",
        "",
        "| Method | RMSE (m) | MAE (m) | P95 (m) |",
        "|---|---:|---:|---:|",
    ]
    for label, est in rows:
        rmse, mae, p95 = metrics(est, data["truth"])
        lines.append(f"| {label} | {rmse:.3f} | {mae:.3f} | {p95:.3f} |")
    lines.append("")
    lines.append("Generated files include PNG and PDF versions for each schematic figure.")
    (OUT_DIR / "README_结果图示意说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    data = generate_synthetic_localization()
    plot_trajectory(data)
    plot_cdf(data)
    plot_metric_bar(data)
    plot_weights(data)
    plot_weight_scatter(data)
    plot_consistency_relationship(data)
    plot_ablation(data)
    plot_snr_robustness(data)
    plot_regularization_effect(data)
    write_summary(data)


if __name__ == "__main__":
    main()
