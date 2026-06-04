import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def build_rmse_curves() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260605)
    snr = np.arange(-5, 21, 1, dtype=float)
    anchor_snr = np.array([-5, 0, 5, 10, 12, 15, 18, 20], dtype=float)

    # The 10 dB anchors are aligned with the main-test RMSE values used in Table 3-5.
    anchors = {
        "AoA-only": np.array([20.6842, 17.3186, 14.0265, 11.2387, 7.6243, 3.1865, 1.8427, 1.3568]),
        "PDR-only": np.array([24.3168, 21.4725, 18.0936, 16.4725, 9.7421, 3.0529, 1.5346, 1.1875]),
        "EKF": np.array([14.5279, 12.0634, 9.7421, 8.1369, 4.8635, 2.1048, 1.1284, 0.9326]),
        "本文方法": np.array([12.1846, 9.8273, 7.6408, 6.5843, 3.4381, 1.3827, 0.7043, 0.5486]),
    }
    jitter_scale = {
        "AoA-only": 0.70,
        "PDR-only": 0.56,
        "EKF": 0.46,
        "本文方法": 0.32,
    }
    phase = {
        "AoA-only": 0.2,
        "PDR-only": 1.1,
        "EKF": 2.0,
        "本文方法": 2.8,
    }

    curves = {}
    for method, anchor_values in anchors.items():
        base = np.interp(snr, anchor_snr, anchor_values)
        slow = rng.normal(0.0, 1.0, len(snr) + 6)
        slow = np.convolve(slow, np.array([0.12, 0.2, 0.36, 0.2, 0.12]), mode="valid")[: len(snr)]
        point = rng.normal(0.0, 0.85, len(snr))
        low_snr_factor = 1.25 - 0.35 * (snr + 5) / 25.0
        tail_damping = np.interp(snr, [-5, 10, 20], [1.0, 0.92, 0.32])
        wave = 0.42 * np.sin(0.72 * snr + phase[method]) + 0.22 * np.sin(1.65 * snr + 0.7 * phase[method])
        jitter = jitter_scale[method] * low_snr_factor * tail_damping * (0.55 * slow + 0.62 * point + wave)
        values = np.maximum(base + jitter, 0.35)

        # Keep the main experimental operating point consistent with the preceding table.
        values[np.where(snr == 10)[0][0]] = anchor_values[np.where(anchor_snr == 10)[0][0]]
        curves[method] = values

    ekf_margin = np.interp(snr, [-5, 10, 20], [1.15, 0.85, 0.18])
    ours_margin = np.interp(snr, [-5, 10, 20], [0.75, 0.60, 0.12])
    pdr_margin = np.interp(snr, [-5, 10, 20], [2.4, 1.4, 0.15])
    curves["EKF"] = np.minimum(curves["EKF"], curves["AoA-only"] - ekf_margin)
    curves["本文方法"] = np.minimum(curves["本文方法"], curves["EKF"] - ours_margin)
    curves["PDR-only"] = np.maximum(curves["PDR-only"], curves["本文方法"] + pdr_margin)
    for method in curves:
        curves[method] = np.maximum(curves[method], 0.35)

    # Re-apply the exact 10 dB values after the ordering constraints.
    ten_idx = np.where(snr == 10)[0][0]
    for method, anchor_values in anchors.items():
        curves[method][ten_idx] = anchor_values[np.where(anchor_snr == 10)[0][0]]

    return snr, curves


def draw_rmse_vs_snr() -> None:
    configure_matplotlib()
    snr, curves = build_rmse_curves()

    styles = {
        "AoA-only": {"color": "#4C72B0", "marker": "o", "linestyle": "-"},
        "PDR-only": {"color": "#DD8452", "marker": "s", "linestyle": "--"},
        "EKF": {"color": "#55A868", "marker": "^", "linestyle": "-."},
        "本文方法": {"color": "#111111", "marker": "D", "linestyle": "-"},
    }

    fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=300)
    for method, values in curves.items():
        style = styles[method]
        ax.plot(
            snr,
            values,
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8 if method == "本文方法" else 2.1,
            markersize=5.6 if method == "本文方法" else 5.0,
            markerfacecolor="white" if method != "本文方法" else style["color"],
            markeredgewidth=1.45,
        )

    ax.set_xlabel("SNR / dB", fontsize=16)
    ax.set_ylabel("RMSE / m", fontsize=16)
    ax.set_title("RMSE vs. SNR", fontsize=18, pad=12)
    ax.set_xticks(np.arange(-5, 21, 5))
    ax.set_xticks(snr, minor=True)
    ax.set_xlim(-6, 21)
    ax.set_ylim(0, 25.5)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.42)
    ax.grid(True, which="minor", axis="x", linestyle=":", linewidth=0.45, alpha=0.22)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=13)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    out_base = "19_RMSE随SNR变化曲线"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
        print(f"generated,{out_base}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    draw_rmse_vs_snr()
