from __future__ import annotations

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
MPL_CONFIG = OUT_DIR / ".mplconfig"
MPL_CONFIG.mkdir(exist_ok=True)
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
    plt.rcParams["font.size"] = 9
    plt.rcParams["savefig.dpi"] = 300


def rounded_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", lw=1.0, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#333333", lw=1.2, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_small_tree(ax, cx, cy, scale=1.0, color="#4A6FA5", title="Tree"):
    ax.text(cx, cy + 0.64 * scale, title, ha="center", va="center", fontsize=8.5, color=color)
    nodes = {
        "root": (cx, cy + 0.38 * scale),
        "left": (cx - 0.30 * scale, cy + 0.12 * scale),
        "right": (cx + 0.30 * scale, cy + 0.12 * scale),
        "ll": (cx - 0.43 * scale, cy - 0.18 * scale),
        "lr": (cx - 0.17 * scale, cy - 0.18 * scale),
        "rl": (cx + 0.17 * scale, cy - 0.18 * scale),
        "rr": (cx + 0.43 * scale, cy - 0.18 * scale),
    }
    edges = [("root", "left"), ("root", "right"), ("left", "ll"), ("left", "lr"), ("right", "rl"), ("right", "rr")]
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=color, lw=1.0)
    for key in ["root", "left", "right"]:
        ax.add_patch(Circle(nodes[key], 0.055 * scale, fc="white", ec=color, lw=1.0))
    for key in ["ll", "lr", "rl", "rr"]:
        rounded_box(ax, nodes[key][0] - 0.09 * scale, nodes[key][1] - 0.055 * scale, 0.18 * scale, 0.11 * scale, "w", fc="#FFFFFF", ec=color, lw=0.8, fontsize=7)
    ax.text(cx, cy - 0.44 * scale, "叶节点输出连续值", ha="center", va="center", fontsize=7.4, color="#555555")


def main() -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(11.2, 6.1))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    ax.text(0.35, 5.82, "XGBoost权重回归器内部结构（以第 i 个权重为例）", fontsize=13, weight="bold", color="#222222")

    input_text = (
        "输入样本\n"
        "$\\mathbf{t}_k\\in\\mathbb{R}^{6}$\n"
        "$[\\sigma_{\\tau,k},\\sigma^2_{RSSI,k},\\bar r_k,d_{12,k},d_{13,k},d_{23,k}]^T$"
    )
    rounded_box(ax, 0.42, 3.68, 2.18, 1.12, input_text, fc="#F7FBFF", ec="#4A6FA5", lw=1.1, fontsize=8.2)
    rounded_box(ax, 0.55, 2.58, 1.92, 0.56, "初始预测\n$\\hat q_{k,i}^{(0)}$", fc="#F4F4F4", ec="#777777", fontsize=8.5)
    arrow(ax, (1.51, 3.65), (1.51, 3.16), color="#777777")

    tree_xs = [3.25, 4.78, 6.30, 7.82]
    tree_titles = ["第1棵树", "第2棵树", "第3棵树", "第M棵树"]
    tree_colors = ["#4A6FA5", "#4C8C62", "#C05B5B", "#8A6BB8"]
    for x, title, color in zip(tree_xs, tree_titles, tree_colors):
        rounded_box(ax, x - 0.62, 3.30, 1.24, 1.62, "", fc="#FFFFFF", ec=color, lw=1.0)
        draw_small_tree(ax, x, 4.02, scale=0.86, color=color, title=title)

    for x1, x2 in zip(tree_xs[:-1], tree_xs[1:]):
        arrow(ax, (x1 + 0.64, 4.10), (x2 - 0.64, 4.10), color="#555555")
    ax.text(7.05, 4.63, "$\\cdots$", fontsize=18, color="#555555", ha="center")
    arrow(ax, (2.62, 4.24), (2.78, 4.24), color="#4A6FA5")
    arrow(ax, (2.48, 2.86), (3.05, 3.29), color="#777777", rad=-0.15)

    ax.text(
        5.50,
        5.18,
        "Boosting思想：后续树不是重新独立预测，而是继续拟合上一轮损失的一阶/二阶梯度，使预测逐步逼近反解权重标签",
        fontsize=8.2,
        color="#555555",
        ha="center",
    )

    sum_text = (
        "加法集成输出\n"
        "$\\hat q_{k,i}^{(M)}=\\hat q_{k,i}^{(0)}+\\eta\\sum_{m=1}^{M} f_{i,m}(\\mathbf{t}_k)$"
    )
    rounded_box(ax, 8.92, 3.68, 1.94, 0.98, sum_text, fc="#F6FBF7", ec="#4C8C62", lw=1.1, fontsize=8.2)
    arrow(ax, (8.46, 4.10), (8.88, 4.10), color="#4C8C62")

    output_text = (
        "连续权重预测\n"
        "$\\hat q_{k,i}^{XGB}\\in\\mathbb{R}$"
    )
    rounded_box(ax, 9.18, 2.42, 1.42, 0.70, output_text, fc="#F7F7F7", ec="#555555", fontsize=8.5)
    arrow(ax, (9.88, 3.64), (9.88, 3.14), color="#555555")

    obj_text = (
        "训练目标\n"
        "$\\mathcal{L}=\\sum_{k=1}^{N} l(q_{k,i}^{*},\\hat q_{k,i})+\\sum_{m=1}^{M}\\Omega(f_{i,m})$\n"
        "$\\Omega(f)=\\gamma T+\\frac{1}{2}\\lambda\\sum_{j=1}^{T}w_j^2$"
    )
    rounded_box(ax, 0.58, 0.70, 3.35, 1.00, obj_text, fc="#FFF9F0", ec="#B87A2A", lw=1.1, fontsize=8.4)

    split_text = (
        "每棵树的节点按特征阈值分裂\n"
        "例如：$d_{13,k}<c_1$，$\\sigma^2_{RSSI,k}>c_2$\n"
        "分裂准则由损失下降量和复杂度惩罚共同决定"
    )
    rounded_box(ax, 4.32, 0.70, 3.10, 1.00, split_text, fc="#F7FBFF", ec="#4A6FA5", lw=1.1, fontsize=8.2)

    multi_text = (
        "三维权重输出的实现\n"
        "对 $q_1^*,q_2^*,q_3^*$ 分别训练同类回归器\n"
        "$f_1(\\mathbf{t}_k),f_2(\\mathbf{t}_k),f_3(\\mathbf{t}_k)$"
    )
    rounded_box(ax, 7.82, 0.70, 2.78, 1.00, multi_text, fc="#F4F7FF", ec="#4A6FA5", lw=1.1, fontsize=8.2)

    ax.text(
        5.58,
        0.25,
        "说明：XGBoost内部不是神经网络层结构，而是由多棵CART回归树组成的加法模型；叶节点输出连续数值，因此适合做融合权重回归。",
        fontsize=8.1,
        color="#555555",
        ha="center",
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_DIR / "11_XGBoost内部结构示意图.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "11_XGBoost内部结构示意图.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
