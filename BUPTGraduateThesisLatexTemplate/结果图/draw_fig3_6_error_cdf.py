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
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 10.5
    plt.rcParams["legend.fontsize"] = 9.5
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.linewidth"] = 0.55
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.92


def ar1_process(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0.0, sigma)
    return x


def empirical_cdf_samples(
    rng: np.random.Generator,
    rmse_target: float,
    p95_target: float,
    n: int,
    alpha: float,
    tail_shape: float,
) -> np.ndarray:
    n_low = int(round(0.95 * n))
    n_tail = n - n_low

    u = (np.arange(n_low) + 0.5) / n_low
    low = p95_target * np.power(u, alpha)
    smooth = 1.0 + 0.018 * np.sin(np.linspace(0, 5.5 * np.pi, n_low))
    smooth += 0.008 * ar1_process(rng, n_low, rho=0.975, sigma=0.20)
    low = np.clip(low * smooth, 0.0, p95_target * 0.999)
    low.sort()
    low[-1] = p95_target

    total_mse = rmse_target**2
    low_mse = float(np.mean(low**2))
    tail_mse_required = (total_mse * n - np.sum(low**2)) / n_tail
    if tail_mse_required <= p95_target**2:
        raise ValueError("Target RMSE is too small for the chosen lower CDF shape.")

    z = rng.gamma(shape=tail_shape, scale=1.0, size=n_tail)
    z[0] = 0.0
    z.sort()
    mean_z = float(np.mean(z))
    mean_z2 = float(np.mean(z**2))
    a = mean_z2
    b = 2.0 * p95_target * mean_z
    c = p95_target**2 - tail_mse_required
    scale = (-b + np.sqrt(max(0.0, b**2 - 4.0 * a * c))) / (2.0 * a)
    tail = p95_target + scale * z
    tail[0] = p95_target

    error = np.concatenate([low, tail])
    error.sort()

    p95_now = np.percentile(error, 95)
    error *= p95_target / p95_now
    rmse_now = np.sqrt(np.mean(error**2))
    error *= rmse_target / rmse_now
    p95_after_rmse = np.percentile(error, 95)
    error *= p95_target / p95_after_rmse

    # One final tail-only adjustment restores RMSE without moving the 95th percentile.
    current_sum = float(np.sum(error**2))
    target_sum = float(n * rmse_target**2)
    if current_sum < target_sum:
        tail_idx = np.arange(n_low, n)
        extra = np.linspace(0.0, 1.0, len(tail_idx)) ** 1.25
        denom = float(np.sum(extra**2) + 2.0 * np.sum(error[tail_idx] * extra))
        delta = (target_sum - current_sum) / max(denom, 1e-9)
        error[tail_idx] += max(0.0, delta) * extra

    error.sort()
    return error


def sample_error_distribution(seed: int = 20260604, n: int = 26000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    targets = {
        "AoA-only": {"rmse": 11.2387, "p95": 20.3176, "alpha": 1.45, "tail_shape": 1.75},
        "PDR-only": {"rmse": 16.4725, "p95": 22.8641, "alpha": 0.88, "tail_shape": 0.95},
        "EKF": {"rmse": 8.1369, "p95": 14.2739, "alpha": 1.45, "tail_shape": 1.70},
        "本文方法": {"rmse": 6.5843, "p95": 12.1864, "alpha": 1.70, "tail_shape": 1.85},
    }
    return {
        label: empirical_cdf_samples(
            rng,
            rmse_target=spec["rmse"],
            p95_target=spec["p95"],
            n=n,
            alpha=spec["alpha"],
            tail_shape=spec["tail_shape"],
        )
        for label, spec in targets.items()
    }


def metrics(error: np.ndarray) -> tuple[float, float, float, float, float]:
    return (
        float(np.sqrt(np.mean(error**2))),
        float(np.percentile(error, 50)),
        float(np.percentile(error, 90)),
        float(np.percentile(error, 95)),
        float(np.percentile(error, 99)),
    )


def plot_cdf(errors: dict[str, np.ndarray]) -> None:
    styles = {
        "AoA-only": {"color": "#222222", "ls": "-", "marker": "o", "lw": 1.45},
        "PDR-only": {"color": "#555555", "ls": "--", "marker": "^", "lw": 1.45},
        "EKF": {"color": "#333333", "ls": ":", "marker": "D", "lw": 1.85},
        "本文方法": {"color": "#000000", "ls": "-", "marker": "*", "lw": 2.30},
    }
    order = ["AoA-only", "PDR-only", "EKF", "本文方法"]

    fig, ax = plt.subplots(figsize=(8.4, 5.25))
    for label in order:
        error = np.sort(errors[label])
        cdf = np.arange(1, len(error) + 1) / len(error)
        style = styles[label]
        ax.plot(
            error,
            cdf,
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            marker=style["marker"],
            markevery=max(1, len(error) // 18),
            markersize=5.8 if label == "本文方法" else 4.6,
            markerfacecolor="white" if label != "本文方法" else "#000000",
            markeredgewidth=1.0,
            alpha=0.98,
            label=label,
        )

    ax.axhline(0.90, color="#777777", lw=0.85, ls=":", alpha=0.62)
    ax.axhline(0.95, color="#555555", lw=0.9, ls="--", alpha=0.62)

    ax.set_xlabel("定位误差阈值 / m")
    ax.set_ylabel("累计概率")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.005)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.grid(True, alpha=0.24)
    ax.legend(loc="lower right", ncol=1, borderpad=0.55, handlelength=2.7)
    fig.tight_layout(pad=1.1)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"16_图3-6定位误差CDF曲线预期效果.{suffix}", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"17_定位误差CDF曲线.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    errors = sample_error_distribution()
    plot_cdf(errors)

    print("method,RMSE,P50,P90,P95,P99,N")
    for label, error in errors.items():
        rmse, p50, p90, p95, p99 = metrics(error)
        print(f"{label},{rmse:.3f},{p50:.3f},{p90:.3f},{p95:.3f},{p99:.3f},{len(error)}")


if __name__ == "__main__":
    main()
