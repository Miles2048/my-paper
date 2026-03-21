from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


# ----------------------------
# 论文场景参数
# ----------------------------
AREA_SIZE_M = 500.0
GRID_SIZE = 90
DISPLAY_RESOLUTION = 260
REFERENCE_DISTANCE_M = 1.0
REFERENCE_LOSS_DB = 31.5
PATH_LOSS_EXPONENT_BASE = 2.95
NOISE_STD_DB_BASE = 3.6
SMOOTH_SIGMA_BASE = 0.72
OUTPUT_DIR = Path("/Users/miles/Desktop/大论文/脚本")


BASE_TARGETS = [
    {"pos": (88.0, 118.0), "tx_power_dbm": 26.0},
    {"pos": (162.0, 356.0), "tx_power_dbm": 24.8},
    {"pos": (286.0, 214.0), "tx_power_dbm": 25.6},
    {"pos": (346.0, 312.0), "tx_power_dbm": 23.9},
    {"pos": (408.0, 154.0), "tx_power_dbm": 25.1},
    {"pos": (430.0, 404.0), "tx_power_dbm": 24.2},
]

BASE_GHOSTS = [
    {"pos": (126.0, 170.0), "peak_dbm": -56.0, "sigma_m": 18.0},
    {"pos": (208.0, 332.0), "peak_dbm": -58.5, "sigma_m": 23.0},
    {"pos": (246.0, 174.0), "peak_dbm": -55.2, "sigma_m": 20.0},
    {"pos": (318.0, 258.0), "peak_dbm": -57.4, "sigma_m": 22.0},
    {"pos": (372.0, 142.0), "peak_dbm": -59.0, "sigma_m": 18.0},
    {"pos": (388.0, 368.0), "peak_dbm": -57.8, "sigma_m": 21.0},
]


VARIANTS = [
    {
        "name": "radiomap_3d_urban_canyon_a",
        "seed": 42,
        "path_loss_exp": 2.90,
        "noise_std": 3.2,
        "smooth_sigma": 0.68,
        "stripe_gain": 1.05,
        "shadow_gain": 1.10,
        "corridor_gain": 0.95,
        "hotspot_gain": 0.26,
        "reflection_gain": 0.34,
        "elev": 35,
        "azim": -124,
    },
    {
        "name": "radiomap_3d_urban_canyon_b",
        "seed": 77,
        "path_loss_exp": 3.02,
        "noise_std": 3.8,
        "smooth_sigma": 0.74,
        "stripe_gain": 1.15,
        "shadow_gain": 1.00,
        "corridor_gain": 1.08,
        "hotspot_gain": 0.30,
        "reflection_gain": 0.40,
        "elev": 37,
        "azim": -127,
    },
    {
        "name": "radiomap_3d_reflection_dense",
        "seed": 103,
        "path_loss_exp": 2.98,
        "noise_std": 4.1,
        "smooth_sigma": 0.70,
        "stripe_gain": 1.20,
        "shadow_gain": 1.18,
        "corridor_gain": 1.02,
        "hotspot_gain": 0.32,
        "reflection_gain": 0.48,
        "elev": 34,
        "azim": -131,
    },
    {
        "name": "radiomap_3d_shadow_fading",
        "seed": 256,
        "path_loss_exp": 3.10,
        "noise_std": 3.9,
        "smooth_sigma": 0.76,
        "stripe_gain": 0.96,
        "shadow_gain": 1.32,
        "corridor_gain": 0.88,
        "hotspot_gain": 0.24,
        "reflection_gain": 0.35,
        "elev": 36,
        "azim": -122,
    },
    {
        "name": "radiomap_3d_multipath_strong",
        "seed": 512,
        "path_loss_exp": 2.87,
        "noise_std": 4.4,
        "smooth_sigma": 0.66,
        "stripe_gain": 1.28,
        "shadow_gain": 1.08,
        "corridor_gain": 1.12,
        "hotspot_gain": 0.34,
        "reflection_gain": 0.52,
        "elev": 38,
        "azim": -129,
    },
]


def dbm_to_mw(power_dbm):
    return 10 ** (power_dbm / 10.0)


def mw_to_dbm(power_mw, eps=1e-12):
    return 10.0 * np.log10(np.maximum(power_mw, eps))


def normalize_field(field):
    field_min = np.min(field)
    field_max = np.max(field)
    if np.isclose(field_min, field_max):
        return np.zeros_like(field)
    return (field - field_min) / (field_max - field_min)


def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def smooth_2d(field, sigma):
    kernel = gaussian_kernel1d(sigma)
    pad = len(kernel) // 2

    padded_x = np.pad(field, ((pad, pad), (0, 0)), mode="edge")
    filtered_x = np.empty_like(field, dtype=float)
    for col in range(field.shape[1]):
        filtered_x[:, col] = np.convolve(padded_x[:, col], kernel, mode="valid")

    padded_y = np.pad(filtered_x, ((0, 0), (pad, pad)), mode="edge")
    filtered = np.empty_like(field, dtype=float)
    for row in range(field.shape[0]):
        filtered[row, :] = np.convolve(padded_y[row, :], kernel, mode="valid")

    return filtered


def bilinear_resize(field, out_h, out_w):
    in_h, in_w = field.shape
    y_old = np.linspace(0.0, 1.0, in_h)
    x_old = np.linspace(0.0, 1.0, in_w)
    y_new = np.linspace(0.0, 1.0, out_h)
    x_new = np.linspace(0.0, 1.0, out_w)

    temp = np.empty((out_h, in_w), dtype=float)
    for j in range(in_w):
        temp[:, j] = np.interp(y_new, y_old, field[:, j])

    resized = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        resized[i, :] = np.interp(x_new, x_old, temp[i, :])

    return resized


def correlated_noise(rng, shape, sigma=1.0):
    raw = rng.normal(0.0, sigma, size=shape)
    return smooth_2d(raw, sigma=2.2)


def add_street_canyon_ripples(xx, yy):
    stripe_a = 3.4 * np.sin(xx / 16.0 + yy / 43.0)
    stripe_b = 2.8 * np.cos(xx / 27.0 - yy / 21.0)
    stripe_c = 1.9 * np.sin((xx + 1.7 * yy) / 52.0)
    return stripe_a + stripe_b + stripe_c


def add_shadow_fading(xx, yy):
    broad_variation = 4.8 * np.sin(xx / 78.0) - 4.2 * np.cos(yy / 66.0)
    diagonal_shadow = -7.8 * np.exp(-((yy - 0.82 * xx - 36.0) ** 2) / (2.0 * 26.0 ** 2))
    blocked_zone = -9.5 * np.exp(
        -(((xx - 352.0) / 52.0) ** 2 + ((yy - 228.0) / 28.0) ** 2) / 2.0
    )
    side_block = -6.2 * np.exp(
        -(((xx - 148.0) / 34.0) ** 2 + ((yy - 404.0) / 44.0) ** 2) / 2.0
    )
    return broad_variation + diagonal_shadow + blocked_zone + side_block


def add_corridor_gain(xx, yy):
    corridor_1 = 5.5 * np.exp(-((yy - 0.58 * xx - 44.0) ** 2) / (2.0 * 18.0 ** 2))
    corridor_2 = 4.3 * np.exp(-((yy + 0.42 * xx - 314.0) ** 2) / (2.0 * 20.0 ** 2))
    return corridor_1 + corridor_2


def add_small_scale_hotspots(rng, xx, yy, count=26):
    hotspot_field = np.zeros_like(xx, dtype=float)
    for _ in range(count):
        cx = rng.uniform(25.0, AREA_SIZE_M - 25.0)
        cy = rng.uniform(25.0, AREA_SIZE_M - 25.0)
        sigma = rng.uniform(8.0, 18.0)
        peak = rng.uniform(-63.0, -55.0)
        distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        hotspot_field += dbm_to_mw(peak - distance_sq / (2.0 * sigma ** 2))
    return mw_to_dbm(hotspot_field)


def add_specular_reflections(xx, yy):
    reflection_field = np.zeros_like(xx, dtype=float)
    lobes = [
        (104.0, 228.0, 20.0, 56.0, -57.0, 24.0),
        (236.0, 118.0, 26.0, 72.0, -58.0, -32.0),
        (302.0, 346.0, 18.0, 64.0, -56.5, 41.0),
        (418.0, 246.0, 22.0, 52.0, -59.0, -18.0),
    ]
    for cx, cy, sx, sy, peak, theta_deg in lobes:
        theta = np.deg2rad(theta_deg)
        x_shift = xx - cx
        y_shift = yy - cy
        x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
        y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
        reflection_field += dbm_to_mw(
            peak - (x_rot ** 2) / (2.0 * sx ** 2) - (y_rot ** 2) / (2.0 * sy ** 2)
        )
    return mw_to_dbm(reflection_field)


def perturb_sources(rng, sources, pos_sigma, power_sigma, sigma_sigma=0.0):
    perturbed = []
    for item in sources:
        new_item = dict(item)
        px, py = item["pos"]
        new_item["pos"] = (
            float(np.clip(px + rng.normal(0.0, pos_sigma), 20.0, AREA_SIZE_M - 20.0)),
            float(np.clip(py + rng.normal(0.0, pos_sigma), 20.0, AREA_SIZE_M - 20.0)),
        )
        if "tx_power_dbm" in item:
            new_item["tx_power_dbm"] = float(item["tx_power_dbm"] + rng.normal(0.0, power_sigma))
        if "peak_dbm" in item:
            new_item["peak_dbm"] = float(item["peak_dbm"] + rng.normal(0.0, power_sigma))
        if "sigma_m" in item:
            new_item["sigma_m"] = float(max(10.0, item["sigma_m"] + rng.normal(0.0, sigma_sigma)))
        perturbed.append(new_item)
    return perturbed


def build_radiomap(variant):
    rng = np.random.default_rng(variant["seed"])

    x = np.linspace(0.0, AREA_SIZE_M, GRID_SIZE)
    y = np.linspace(0.0, AREA_SIZE_M, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)

    targets = perturb_sources(rng, BASE_TARGETS, pos_sigma=9.0, power_sigma=0.8)
    ghosts = perturb_sources(rng, BASE_GHOSTS, pos_sigma=12.0, power_sigma=1.0, sigma_sigma=2.6)
    total_power_mw = np.zeros_like(xx, dtype=float)

    for target in targets:
        tx_x, tx_y = target["pos"]
        distance = np.sqrt((xx - tx_x) ** 2 + (yy - tx_y) ** 2)
        distance = np.maximum(distance, REFERENCE_DISTANCE_M)
        path_loss_db = (
            REFERENCE_LOSS_DB
            + 10.0 * variant["path_loss_exp"] * np.log10(distance / REFERENCE_DISTANCE_M)
        )
        received_dbm = target["tx_power_dbm"] - path_loss_db
        total_power_mw += dbm_to_mw(received_dbm)

    for ghost in ghosts:
        gx, gy = ghost["pos"]
        sigma = ghost["sigma_m"]
        distance_sq = (xx - gx) ** 2 + (yy - gy) ** 2
        ghost_dbm = ghost["peak_dbm"] - distance_sq / (2.0 * sigma ** 2)
        total_power_mw += dbm_to_mw(ghost_dbm)

    base_power_dbm = mw_to_dbm(total_power_mw)
    ripple_db = add_street_canyon_ripples(xx, yy)
    shadow_db = add_shadow_fading(xx, yy)
    corridor_db = add_corridor_gain(xx, yy)
    hotspot_db = add_small_scale_hotspots(rng, xx, yy)
    reflection_db = add_specular_reflections(xx, yy)
    large_scale_noise = correlated_noise(rng, xx.shape, sigma=2.4)
    small_scale_noise = rng.normal(loc=0.0, scale=variant["noise_std"], size=xx.shape)

    total_power_dbm = (
        base_power_dbm
        + variant["stripe_gain"] * ripple_db
        + variant["shadow_gain"] * shadow_db
        + variant["corridor_gain"] * corridor_db
        + variant["hotspot_gain"] * hotspot_db
        + variant["reflection_gain"] * reflection_db
        + large_scale_noise
        + small_scale_noise
    )

    smoothed_dbm = smooth_2d(total_power_dbm, sigma=variant["smooth_sigma"])
    display_map = bilinear_resize(smoothed_dbm, DISPLAY_RESOLUTION, DISPLAY_RESOLUTION)
    return display_map


def save_radiomap_figure(variant):
    display_map = build_radiomap(variant)
    display_map_norm = normalize_field(display_map)
    z_map = -102.0 + 49.0 * display_map_norm

    x_fine = np.linspace(0.0, AREA_SIZE_M, DISPLAY_RESOLUTION)
    y_fine = np.linspace(0.0, AREA_SIZE_M, DISPLAY_RESOLUTION)
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)

    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(10.8, 8.0), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    facecolors = cm.Greys(0.34 + 0.36 * display_map_norm)
    ax.plot_surface(
        xx_fine,
        yy_fine,
        z_map,
        facecolors=facecolors,
        linewidth=0.10,
        edgecolor=(0.35, 0.35, 0.35, 0.12),
        antialiased=True,
        shade=True,
        alpha=0.99,
        rcount=220,
        ccount=220,
    )

    ax.set_title("Three-Dimensional Radio Map", pad=12, fontsize=15)
    ax.set_xlabel("X / m", labelpad=10)
    ax.set_ylabel("Y / m", labelpad=10)
    ax.set_zlabel("RSSI / dBm", labelpad=8)
    ax.set_xlim(0.0, AREA_SIZE_M)
    ax.set_ylim(0.0, AREA_SIZE_M)
    ax.set_zlim(np.min(z_map) - 1.5, np.max(z_map) + 1.5)
    ax.view_init(elev=variant["elev"], azim=variant["azim"])
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))
    ax.grid(False)

    output_path = OUTPUT_DIR / f"{variant['name']}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = [save_radiomap_figure(variant) for variant in VARIANTS]
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
