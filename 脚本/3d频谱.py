import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor

# 参数设置
N, M, T = 100, 100, 10  # 空间尺寸和时间步长
K = 10 # 团聚中心数量
cluster_size_range = (10, 20)  # 团聚尺寸范围
source_intensity_range=[4,10]
snr_db=20

def normalize_pair_minmax(a, b, eps=1e-12):
    """Use a shared min-max range to normalize two arrays to [0, 1]."""
    vmin = min(np.min(a), np.min(b))
    vmax = max(np.max(a), np.max(b))
    scale = vmax - vmin
    if scale < eps:
        return np.zeros_like(a), np.zeros_like(b)
    return (a - vmin) / scale, (b - vmin) / scale

def calculate_sre(reference, reconstructed, eps=1e-12):
    """SRE = ||X_hat - X||^2 / ||X||^2."""
    signal_energy = np.sum(reference ** 2)
    error_energy = np.sum((reference - reconstructed) ** 2)
    return error_energy / (signal_energy + eps)

# 生成原始信号
# def generate_aggregated_signal(N, M, T, K, cluster_size_range):
#     """
#     生成具有团聚特点的三维稀疏信号
#     """
#     x = np.zeros((N, M, T))
#     for _ in range(K):
#         center_x = np.random.randint(0, N)
#         center_y = np.random.randint(0, M)
#         center_t = np.random.randint(0, T)
#         cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
#         sigma = cluster_size / 3
#         for i in range(N):
#             for j in range(M):
#                 for k in range(T):
#                     distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2 + (k - center_t) ** 2)
#                     x[i, j, k] += np.exp(-distance ** 2 / (2 * sigma ** 2))
#     return x

def generate_aggregated_signal(N, M, T, K, cluster_size_range):
    """生成具有团聚特点的空间信号，时间上呈方波特性"""

    x = np.zeros((N, M, T))

    # 生成随机的团簇中心，团簇大小，以及方波的开启/关闭时间
    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        sigma = cluster_size / 3  # 控制团簇的扩散程度

        on_off_time = np.random.randint(0, T)  # 随机选取方波开启/关闭时间

        # 根据高斯分布在空间上分布团簇
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                spatial_influence = np.exp(-distance ** 2 / (2 * sigma ** 2))

                # 在时间上，以方波形式开关信号
                for t in range(T):
                    # t < on_off_time 时，信号关闭(0)；t >= on_off_time 时，信号开启(高斯幅值)
                    if t >= on_off_time:
                        x[i, j, t] += spatial_influence

    return x
def generate_clustered_signal(N, M, T, K, cluster_size_range, source_intensity_range, snr_db):
    """
    生成具有团聚特点的空间信号，时间上呈方波特性，并加入可控信噪比的噪声。

    Args:
        N, M, T: 信号的空间和时间维度。
        K: 团簇的数量。
        cluster_size_range: 团簇大小的范围 (min, max)。
        source_intensity_range: 辐射源中心强度范围 (min, max)。
        snr_db: 信噪比，单位为 dB，基于最大辐射源中心强度。

    Returns:
        x: 生成的三维信号。
    """

    x = np.zeros((N, M, T))
    max_source_intensity = 0  # 用于跟踪最大的辐射源强度

    # 生成随机的团簇中心，团簇大小，以及方波的开启/关闭时间
    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        source_intensity = np.random.uniform(source_intensity_range[0], source_intensity_range[1]) # 辐射源强度

        sigma = cluster_size / 3  # 控制团簇的扩散程度

        on_off_time = np.random.randint(0, T)  # 随机选取方波开启/关闭时间

        # 根据高斯分布在空间上分布团簇
        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                spatial_influence = np.exp(-distance ** 2 / (2 * sigma ** 2))

                # 在时间上, 以方波形式开关信号
                for t in range(T):
                    # t < on_off_time 时，信号关闭(0)；t >= on_off_time 时，信号开启 (高斯幅值 * 信号强度)
                    if t >= on_off_time:
                        x[i, j, t] += spatial_influence * source_intensity

        max_source_intensity = max(max_source_intensity, source_intensity) #更新最大值

    # Calculate noise power based on maximum source intensity
    signal_power = max_source_intensity ** 2  # Power is proportional to the square of intensity
    snr_linear = 10 ** (snr_db / 10)  # convert dB to linear scale
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, size=(N, M, T)) # Gaussian noise

    # Add noise to the signal
    x_noisy = x + noise

    return x_noisy

original_signal = generate_clustered_signal(N, M, T, K, cluster_size_range,source_intensity_range, snr_db)

# 随机采样 16% 的数据
sample_rate = 0.05
Sample_Location = np.random.choice(N * M * T, size=int(N * M * T * sample_rate), replace=False)
Sample_Location_3D = np.unravel_index(Sample_Location, (N, M, T))
sample_coords = np.array(Sample_Location_3D).T  # 形状为 (n_samples, 3)
sample_values = original_signal.ravel()[Sample_Location]

# 使用随机森林进行插值
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(sample_coords, sample_values)

# 生成查询网格（完整三维网格）
x, y, t = np.arange(N), np.arange(M), np.arange(T)
grid_x, grid_y, grid_t = np.meshgrid(x, y, t, indexing='ij')
query_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_t.ravel()))

# 预测插值结果
predicted_values = model.predict(query_points)
predicted_array = predicted_values.reshape(N, M, T)

# 对原始数据和恢复数据做归一化
original_signal_norm, predicted_array_norm = normalize_pair_minmax(original_signal, predicted_array)

# 用归一化后的数据计算 SRE：||X_hat - X||^2 / ||X||^2
sre = calculate_sre(original_signal_norm, predicted_array_norm)
print(f"SRE (normalized): {sre:.6f}")

# 创建三维可视化
fig = plt.figure(figsize=(18, 6))

# 原始信号
ax1 = fig.add_subplot(131, projection='3d')
X, Y = np.meshgrid(np.arange(N), np.arange(M))
Z = original_signal_norm[:, :, T//2]  # 选择中间时间步
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('Original Signal (Normalized)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Signal Value')

# 采样信号
ax2 = fig.add_subplot(132, projection='3d')
sample_x, sample_y, sample_t = Sample_Location_3D
mask = sample_t == T//2  # 选择中间时间步
ax2.scatter(sample_x[mask], sample_y[mask], original_signal_norm[sample_x[mask], sample_y[mask], sample_t[mask]], c='red', s=50)
ax2.set_title('Sampled Signal (Normalized)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Signal Value')

# 恢复信号
ax3 = fig.add_subplot(133, projection='3d')
Z_recovered = predicted_array_norm[:, :, T//2]  # 选择中间时间步
ax3.plot_surface(X, Y, Z_recovered, cmap='viridis', edgecolor='none')
ax3.set_title('Recovered Signal (Normalized)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Signal Value')

# 添加采样率和 SRE 信息
plt.suptitle(f"Sample Rate: {sample_rate} | SRE: {sre:.6f}", fontsize=16)

plt.tight_layout()
plt.show()
