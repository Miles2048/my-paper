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
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name, "Arial", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["savefig.dpi"] = 450


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    fc: str = "#FFFFFF",
    ec: str = "#333333",
    lw: float = 1.2,
    fontsize: float = 10,
    radius: float = 0.07,
    weight: str | None = None,
    ls: str = "-",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.28,
            weight=weight,
        )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#222222",
    lw: float = 1.5,
    ls: str = "-",
    rad: float = 0.0,
    scale: float = 13,
) -> None:
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


def draw_tree(ax, x: float, y: float, w: float, h: float, title: str, color: str) -> None:
    add_box(ax, x, y, w, h, fc="#FFFFFF", ec=color, lw=1.1, radius=0.055)
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=10.2, weight="bold")

    cx = x + w / 2
    root = (cx, y + h - 0.62)
    l1 = (cx - 0.34, y + h - 1.02)
    r1 = (cx + 0.34, y + h - 1.02)
    leaves = [
        (cx - 0.55, y + h - 1.38, "$w_1$"),
        (cx - 0.22, y + h - 1.38, "$w_2$"),
        (cx + 0.22, y + h - 1.38, "$\\cdots$"),
        (cx + 0.55, y + h - 1.38, "$w_T$"),
    ]
    for a, b in [(root, l1), (root, r1), (l1, leaves[0][:2]), (l1, leaves[1][:2]), (r1, leaves[2][:2]), (r1, leaves[3][:2])]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#111111", lw=1.0)
    for p in [root, l1, r1]:
        ax.add_patch(Circle(p, 0.105, fc="#B7D4EE", ec="#111111", lw=0.8))
    for lx, ly, label in leaves:
        add_box(ax, lx - 0.145, ly - 0.105, 0.29, 0.21, text=label, fc="#EEF5FF", ec=color, lw=0.8, fontsize=8.8, radius=0.03)
    ax.text(x + w / 2, y + 0.14, "叶节点输出连续修正值", ha="center", va="center", fontsize=8.2, color="#333333")


def main() -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(14.8, 8.4))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    blue = "#2F6EA8"
    pale_blue = "#EFF6FF"
    pale_orange = "#FFF0E0"
    orange = "#C4661A"
    pale_green = "#F2FAED"
    green = "#5A8F3D"
    pale_gray = "#F7F7F7"

    input_text = (
        "6维特征输入\n\n"
        "$\\mathbf{t}_k=[\\sigma_{\\tau,k},\\sigma^2_{RSSI,k},\\bar r_k,$\n"
        "$d_{12,k},d_{13,k},d_{23,k}]^T$\n\n"
        "$\\mathbf{t}_k\\in\\mathbb{R}^{6}$"
    )
    add_box(ax, 0.25, 3.08, 2.35, 2.55, input_text, fc=pale_blue, ec=blue, lw=1.3, fontsize=11.2, weight="bold")

    label_text = (
        "离线反解权重标签\n"
        "$\\mathbf{q}_k^*=[q_{k,1}^*,q_{k,2}^*,q_{k,3}^*]^T$\n"
        "$\\mathbf{q}_k^*\\in\\mathbb{R}^{3}$"
    )
    add_box(ax, 3.05, 6.75, 3.35, 1.18, label_text, fc=pale_orange, ec=orange, lw=1.35, fontsize=11.0, weight="bold")

    add_box(ax, 2.95, 1.60, 8.95, 4.52, fc="#F3F8FF", ec=blue, lw=1.35, radius=0.075)
    ax.text(7.42, 5.83, "XGBoost回归器（以第 $i$ 个权重为例）", ha="center", va="center", fontsize=13.2, weight="bold")
    add_box(
        ax,
        4.25,
        5.23,
        6.18,
        0.44,
        "逐轮拟合损失梯度，后续树修正前一轮预测残差",
        fc="#FFFFFF",
        ec=blue,
        lw=1.0,
        fontsize=10.2,
        radius=0.035,
        ls="--",
    )

    tree_y = 3.15
    tree_w = 1.72
    tree_h = 1.74
    xs = [3.25, 5.30, 7.35, 10.05]
    titles = ["Tree 1", "Tree 2", "Tree 3", "Tree $M$"]
    for x, title in zip(xs, titles):
        draw_tree(ax, x, tree_y, tree_w, tree_h, title, blue)
    add_arrow(ax, (4.98, 4.02), (5.27, 4.02), lw=1.3)
    add_arrow(ax, (7.03, 4.02), (7.32, 4.02), lw=1.3)
    add_arrow(ax, (9.08, 4.02), (9.98, 4.02), lw=1.3)
    ax.text(9.46, 4.10, "$\\cdots$", fontsize=18, ha="center", va="center")

    add_box(
        ax,
        3.35,
        1.88,
        8.15,
        0.70,
        "$\\hat q_{k,i}^{XGB}=\\hat q_{k,i}^{(0)}+\\eta\\sum_{m=1}^{M} f_{i,m}(\\mathbf{t}_k)$",
        fc="#FFFFFF",
        ec=blue,
        lw=1.0,
        fontsize=13.2,
        radius=0.045,
        ls="--",
    )

    output_text = (
        "预测融合权重\n"
        "$\\hat{\\mathbf{q}}_k=[\\hat q_{k,1},\\hat q_{k,2},\\hat q_{k,3}]^T$\n"
        "$\\hat{\\mathbf{q}}_k\\in\\mathbb{R}^{3}$"
    )
    norm_text = (
        "仿射归一化\n"
        "$\\mathbf{1}^T\\hat{\\mathbf{q}}_k=1$\n"
        "不强制 $\\hat q_{k,i}\\geq0$"
    )
    fusion_text = (
        "位置级融合\n"
        "$\\hat{\\mathbf{p}}_k^{Fused}=\\mathbf{P}_k\\hat{\\mathbf{q}}_k$\n"
        "$\\hat{\\mathbf{p}}_k^{Fused}\\in\\mathbb{R}^{2}$"
    )
    add_box(ax, 12.35, 5.00, 2.12, 1.50, output_text, fc=pale_blue, ec=blue, lw=1.3, fontsize=10.8, weight="bold")
    add_box(ax, 12.35, 3.35, 2.12, 1.22, norm_text, fc=pale_green, ec=green, lw=1.25, fontsize=10.8, weight="bold")
    add_box(ax, 12.35, 1.70, 2.12, 1.18, fusion_text, fc=pale_gray, ec="#555555", lw=1.25, fontsize=10.6, weight="bold")

    loss_text = (
        "目标函数\n"
        "$L=\\sum_{k=1}^{N}l(\\mathbf{q}_k^*,\\hat{\\mathbf{q}}_k)+\\sum_{m=1}^{M}\\Omega(f_m)$\n"
        "$l(\\cdot,\\cdot)$：损失函数（如平方损失）"
    )
    reg_text = (
        "复杂度正则\n"
        "$\\Omega(f)=\\gamma T+\\frac{1}{2}\\lambda\\sum_{j=1}^{T}w_j^2$\n"
        "$T$：叶节点数，$w_j$：第 $j$ 个叶节点输出值"
    )
    add_box(ax, 2.86, 0.28, 4.15, 1.24, loss_text, fc=pale_green, ec=green, lw=1.25, fontsize=10.1, weight="bold")
    add_box(ax, 7.55, 0.28, 4.15, 1.24, reg_text, fc=pale_green, ec=green, lw=1.25, fontsize=10.1, weight="bold")

    add_arrow(ax, (2.60, 4.34), (2.95, 4.34), lw=1.6)
    add_arrow(ax, (4.72, 6.74), (7.48, 6.14), lw=1.2, ls="--")
    add_arrow(ax, (11.90, 4.32), (12.34, 5.75), lw=1.5)
    add_arrow(ax, (13.41, 5.00), (13.41, 4.58), lw=1.5)
    add_arrow(ax, (13.41, 3.35), (13.41, 2.88), lw=1.5)
    add_arrow(ax, (7.42, 1.88), (5.05, 1.52), lw=1.2, ls="--")
    add_arrow(ax, (7.42, 1.88), (9.62, 1.52), lw=1.2, ls="--")

    fig.tight_layout(pad=0.25)

    for path in [
        OUT_DIR / "10_XGBoost权重回归结构图.png",
        FIG_DIR / "15_XGBoost权重回归结构图.png",
    ]:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    for path in [
        OUT_DIR / "10_XGBoost权重回归结构图.pdf",
        FIG_DIR / "15_XGBoost权重回归结构图.pdf",
    ]:
        fig.savefig(path, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
