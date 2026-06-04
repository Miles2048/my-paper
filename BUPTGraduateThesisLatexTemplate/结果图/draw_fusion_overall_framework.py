from __future__ import annotations

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = OUT_DIR.parent
FIG_DIR = PROJECT_DIR / "figures" / "ch3"
MPL_CONFIG = OUT_DIR / ".mplconfig"
MPL_CONFIG.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


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
    plt.rcParams["font.sans-serif"] = [font_name or "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["savefig.dpi"] = 450


def box(ax, x, y, w, h, text, fc, ec, fontsize=11, weight=None, lw=1.25, ls="-"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#222222", lw=1.45, ls="-", rad=0.0, scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            linestyle=ls,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def tree_icon(ax, cx, cy, color):
    pts = [(cx, cy + 0.18), (cx - 0.16, cy - 0.08), (cx + 0.16, cy - 0.08)]
    for a, b in [(pts[0], pts[1]), (pts[0], pts[2])]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.0)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=38, color="#DCECF8", edgecolor=color, linewidth=0.9, zorder=3)


def main() -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    blue = "#2E6EA7"
    dark_blue = "#174B77"
    green = "#4D9859"
    teal = "#2A9187"
    orange = "#C96C1B"
    gray = "#59626B"
    pale_blue = "#EAF4FF"
    pale_green = "#EDF8EE"
    pale_orange = "#FFF0E1"
    pale_gray = "#F8F8F8"

    # Left localization modules, same logic as the old framework.
    box(ax, 0.42, 5.25, 1.28, 0.58, "RSSI\n模块", "#144C73", "#144C73", fontsize=10.8, weight="bold")
    box(ax, 0.42, 4.15, 1.28, 0.58, "AoA\n模块", teal, teal, fontsize=10.8, weight="bold")
    box(ax, 0.42, 3.05, 1.28, 0.58, "轨迹递推\n模块", green, green, fontsize=10.2, weight="bold")
    for y in [5.54, 4.44, 3.34]:
        ax.text(1.82, y + 0.08, "原始数据", fontsize=8.8, color="#333333", ha="left", va="center")
        arrow(ax, (1.70, y), (2.36, y), dark_blue, lw=1.1)

    # Feature extractor.
    box(ax, 2.42, 2.62, 1.82, 3.30, "", "#6F8898", "#4A6575", lw=1.3)
    ax.text(3.33, 5.56, "物理感知与\n一致性特征提取器", ha="center", va="center", fontsize=10.8, weight="bold", color="white")
    box(ax, 2.62, 4.88, 1.42, 0.44, "RMS 时延扩展\n$\\sigma_{\\tau,k}$", "#FFFFFF", "#FFFFFF", fontsize=8.8)
    box(ax, 2.62, 4.20, 1.42, 0.44, "RSSI 方差\n$\\sigma^2_{RSSI,k}$", "#FFFFFF", "#FFFFFF", fontsize=8.8)
    box(ax, 2.62, 3.52, 1.42, 0.44, "平均 RSSI\n$\\bar r_k$", "#FFFFFF", "#FFFFFF", fontsize=8.8)
    box(ax, 2.62, 2.84, 1.42, 0.44, "一致性距离\n$d_{12,k},d_{13,k},d_{23,k}$", "#FFFFFF", "#FFFFFF", fontsize=8.2)

    # Position output lines on the top, resembling the reference figure.
    source_lines = [
        (5.83, "$\\mathbf{p}_{k}^{RSSI}$", "#556A7C"),
        (5.62, "$\\mathbf{p}_{k}^{AoA}$", teal),
        (5.41, "$\\mathbf{p}_{k}^{TR}$", green),
    ]
    for y, label, color in source_lines:
        ax.plot([1.70, 10.55], [y, y], color=color, lw=1.0)
        ax.text(8.10, y + 0.03, label, fontsize=8.9, color="#111111", ha="left", va="bottom")

    # XGBoost supervised weight learner replaces the PA-DQN agent in the old figure.
    box(ax, 5.15, 3.18, 2.75, 1.78, "", pale_blue, blue, lw=1.25)
    ax.text(6.52, 4.62, "XGBoost 权重回归器", fontsize=11.2, weight="bold", ha="center")
    ax.text(6.52, 4.33, "监督学习：特征 $\\rightarrow$ 权重", fontsize=9.4, ha="center", color="#333333")
    for i, (x, label, color) in enumerate([(5.72, "$f_1$", blue), (6.52, "$f_2$", teal), (7.32, "$f_3$", orange)]):
        tree_icon(ax, x, 3.76, color)
        ax.text(x, 3.38, label, fontsize=10.2, ha="center", weight="bold")
    ax.text(6.52, 3.18, "$\\hat{\\mathbf{q}}_k=f_{XGB}(\\mathbf{t}_k)$", fontsize=10.8, ha="center", va="bottom")

    # Feature vector arrow.
    ax.text(4.38, 4.12, "6维特征\n$\\mathbf{t}_k$", fontsize=9.2, ha="center", va="center")
    arrow(ax, (4.26, 4.10), (5.12, 4.10), dark_blue, lw=1.2)

    # Offline supervised label branch.
    box(ax, 5.12, 1.20, 2.10, 0.72, "离线反解权重标签\n$\\mathbf{q}_k^*$", pale_orange, orange, fontsize=9.8, weight="bold")
    box(ax, 2.70, 1.20, 1.62, 0.72, "真实位置\n$\\mathbf{p}_k^{th}$", pale_orange, orange, fontsize=9.8, weight="bold")
    arrow(ax, (4.34, 1.56), (5.10, 1.56), orange, lw=1.15, ls="--")
    arrow(ax, (6.16, 1.94), (6.42, 3.15), orange, lw=1.15, ls="--")
    ax.text(6.90, 2.33, "离线监督训练", fontsize=9.0, color=orange, ha="center")

    # Fusion center and output.
    ax.add_patch(Circle((9.45, 4.18), 0.32, facecolor="#FFE3C4", edgecolor=orange, linewidth=1.2))
    ax.text(9.45, 4.18, "$\\Sigma$", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(9.58, 4.90, "自适应加权\n融合中心", ha="left", va="center", fontsize=9.8, weight="bold")
    arrow(ax, (7.92, 4.10), (9.12, 4.18), dark_blue, lw=1.25)
    ax.text(8.25, 4.32, "融合权重\n$\\hat{\\mathbf{q}}_k$", fontsize=9.0, ha="left", va="bottom")
    for y, _, color in source_lines:
        ax.plot([10.55, 10.55], [y, 4.42], color=color, lw=1.0)
        arrow(ax, (10.55, 4.42), (9.76, 4.30), color, lw=1.0)
    arrow(ax, (9.77, 4.18), (11.12, 4.18), dark_blue, lw=1.35)
    ax.text(10.26, 4.36, "$\\hat{\\mathbf{p}}_k^{Fused}$", fontsize=9.4, ha="left")
    box(ax, 11.16, 3.65, 1.10, 0.92, "高精度\n位置估计", "#FFFFFF", gray, fontsize=10.0, weight="bold")

    # Title/caption-like text inside the image is avoided; LaTeX caption will provide title.
    fig.tight_layout(pad=0.2)
    outputs = [
        OUT_DIR / "13_监督权重学习融合定位总体框架图.png",
        FIG_DIR / "2_监督权重学习融合定位总体框架图.png",
    ]
    pdf_outputs = [
        OUT_DIR / "13_监督权重学习融合定位总体框架图.pdf",
        FIG_DIR / "2_监督权重学习融合定位总体框架图.pdf",
    ]
    for path in outputs:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    for path in pdf_outputs:
        fig.savefig(path, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
