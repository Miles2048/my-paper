import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
import os

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

def generate_clustered_signal(N, M, T, K, cluster_size_range, source_intensity_range, snr_db):
    """
    生成具有团聚特点的空间信号，时间上呈方波特性，并加入可控信噪比的噪声。
    """
    x = np.zeros((N, M, T))
    max_source_intensity = 0  # 用于跟踪最大的辐射源强度

    for _ in range(K):
        center_x = np.random.randint(0, N)
        center_y = np.random.randint(0, M)
        cluster_size = np.random.randint(cluster_size_range[0], cluster_size_range[1])
        source_intensity = np.random.uniform(source_intensity_range[0], source_intensity_range[1]) # 辐射源强度

        sigma = cluster_size / 3  # 控制团簇的扩散程度
        on_off_time = np.random.randint(0, T)  # 随机选取方波开启/关闭时间

        for i in range(N):
            for j in range(M):
                distance = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                spatial_influence = np.exp(-distance ** 2 / (2 * sigma ** 2))
                for t in range(T):
                    if t >= on_off_time:
                        x[i, j, t] += spatial_influence * source_intensity
        max_source_intensity = max(max_source_intensity, source_intensity)

    signal_power = max_source_intensity ** 2
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, size=(N, M, T))
    x_noisy = x + noise
    return x_noisy

original_signal = generate_clustered_signal(N, M, T, K, cluster_size_range,source_intensity_range, snr_db)

# 随机采样
sample_rate = 0.05
Sample_Location = np.random.choice(N * M * T, size=int(N * M * T * sample_rate), replace=False)
Sample_Location_3D = np.unravel_index(Sample_Location, (N, M, T))
sample_coords = np.array(Sample_Location_3D).T
sample_values = original_signal.ravel()[Sample_Location]

# 使用随机森林进行插值
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(sample_coords, sample_values)

# 生成查询网格
x, y, t = np.arange(N), np.arange(M), np.arange(T)
grid_x, grid_y, grid_t = np.meshgrid(x, y, t, indexing='ij')
query_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_t.ravel()))

# 预测插值结果
predicted_values = model.predict(query_points)
predicted_array = predicted_values.reshape(N, M, T)

# 归一化
original_signal_norm, predicted_array_norm = normalize_pair_minmax(original_signal, predicted_array)

# 计算 SRE
sre = calculate_sre(original_signal_norm, predicted_array_norm)
print(f"SRE (normalized): {sre:.6f}")

# 创建三维可视化
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
X, Y = np.meshgrid(np.arange(N), np.arange(M))
Z = original_signal_norm[:, :, T//2]
ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax1.set_title('Original Signal (Normalized)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Signal Value')

ax2 = fig.add_subplot(132, projection='3d')
sample_x, sample_y, sample_t = Sample_Location_3D
mask = sample_t == T//2
ax2.scatter(sample_x[mask], sample_y[mask], original_signal_norm[sample_x[mask], sample_y[mask], sample_t[mask]], c='red', s=50)
ax2.set_title('Sampled Signal (Normalized)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Signal Value')

ax3 = fig.add_subplot(133, projection='3d')
Z_recovered = predicted_array_norm[:, :, T//2]
ax3.plot_surface(X, Y, Z_recovered, cmap='viridis', edgecolor='none')
ax3.set_title('Recovered Signal (Normalized)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Signal Value')

plt.suptitle(f"Sample Rate: {sample_rate} | SRE: {sre:.6f}", fontsize=16)
plt.tight_layout()

# 保存图片
output_path = 'BUPTGraduateThesisLatexTemplate/figures/ch5/radiomap_reconstruction.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Figure saved to {output_path}")
