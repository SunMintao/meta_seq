import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'


# ------------------------------
# 1. 数据加载与分组
# ------------------------------
def load_and_group_data(file_path):
    """
    读取Excel文件，返回每个代谢物的6个处理组均值（原始丰度）
    以及每个代谢物所属的类别。
    """
    df = pd.read_excel(file_path, sheet_name=0, header=0)
    compounds = df.iloc[:, 0].values
    classes = df.iloc[:, 1].values
    sample_data = df.iloc[:, 2:]  # 样品列

    # 解析样品名 -> (时期, 基因型, 重复)
    sample_info = []
    for col in sample_data.columns:
        parts = col.split('-')
        # 样品名格式: "G-lht1-1" -> period='G', genotype='lht1'
        if len(parts) >= 2:
            period = parts[0]  # G 或 R
            genotype = parts[1]  # lht1, WT, OX
        else:
            raise ValueError(f"样品名格式错误: {col}")
        sample_info.append((period, genotype))

    # 为每个代谢物计算 (时期, 基因型) 组合的均值
    # 定义处理组顺序
    periods = ['G', 'R']
    genotypes = ['lht1', 'WT', 'OX']
    group_names = [f"{p}-{g}" for p in periods for g in genotypes]  # ['G-lht1', 'G-WT', 'G-OX', ...]

    # 存储每个代谢物的6个均值
    all_means = []
    for idx in range(sample_data.shape[0]):
        values = sample_data.iloc[idx, :].values
        # 创建一个字典，存储每个(period, genotype)的均值
        mean_dict = {}
        for (period, genotype), val in zip(sample_info, values):
            key = f"{period}-{genotype}"
            mean_dict.setdefault(key, []).append(val)
        # 计算均值
        means = [np.mean(mean_dict[key]) for key in group_names]
        all_means.append(means)

    means_df = pd.DataFrame(all_means, index=compounds, columns=group_names)
    return means_df, pd.Series(classes, index=compounds)


# ------------------------------
# 2. 标准化（每个代谢物跨6个处理组Z-score）
# ------------------------------
def standardize_across_groups(means_df):
    """对每个代谢物，在6个处理组内进行Z-score标准化"""
    # 注意：axis=1 表示按行标准化
    return means_df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)


# ------------------------------
# 3. 计算某一类代谢物的整体趋势（均值±SEM）
# ------------------------------
def class_trend(class_name, means_df_std, class_series, periods, genotypes):
    """
    返回该类代谢物标准化后的平均值和标准误，按时期和基因型整理。
    """
    # 筛选属于该类别的代谢物
    mask = class_series == class_name
    if mask.sum() == 0:
        return None
    sub_df = means_df_std.loc[mask]

    # 计算每个处理组的均值和标准误
    group_names = [f"{p}-{g}" for p in periods for g in genotypes]
    mean_vals = sub_df[group_names].mean(axis=0).values
    sem_vals = sub_df[group_names].sem(axis=0).values

    # 重塑为 2x3 矩阵 (时期 x 基因型)
    mean_by_period = {p: {} for p in periods}
    sem_by_period = {p: {} for p in periods}
    for i, p in enumerate(periods):
        for j, g in enumerate(genotypes):
            idx = i * 3 + j
            mean_by_period[p][g] = mean_vals[idx]
            sem_by_period[p][g] = sem_vals[idx]
    return mean_by_period, sem_by_period


# ------------------------------
# 4. 绘制单个类别的趋势图及线性拟合
# ------------------------------
def plot_class_trend(class_name, mean_by_period, sem_by_period,
                     output_dir, periods=['G', 'R'], genotypes=['lht1', 'WT', 'OX']):
    """
    绘制两个时期的剂量‑响应曲线，并添加线性拟合和R²。
    """
    # 剂量水平编码: lht1=-1, WT=0, OX=1
    dose_map = {'lht1': -1, 'WT': 0, 'OX': 1}
    doses = [dose_map[g] for g in genotypes]

    plt.figure(figsize=(7, 5))
    colors = {'G': 'blue', 'R': 'red'}
    markers = {'G': 'o', 'R': 's'}

    for period in periods:
        means = [mean_by_period[period][g] for g in genotypes]
        sem = [sem_by_period[period][g] for g in genotypes]

        # 绘制点 + 误差棒
        plt.errorbar(doses, means, yerr=sem, fmt=markers[period],
                     color=colors[period], capsize=5, label=f'{period} period')

        # 线性回归拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, means)
        x_fit = np.linspace(-1, 1, 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, '--', color=colors[period], alpha=0.6,
                 label=f'{period} fit: R²={r_value ** 2:.2f}, p={p_value:.2e}')

    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Iht1 dosage (lht1 deficiency -> WT -> overexpression)')
    plt.ylabel('Standardized abundance (Z-score)')
    plt.title(f'{class_name}\nLinear trend across Iht1 dosage (G vs R period)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    safe_name = class_name.replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"{safe_name}_trend.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存图片: {out_path}")


# ------------------------------
# 5. 主流程：对所有类别进行建模
# ------------------------------
def build_trend_models(file_path, output_dir="class_trend_plots"):
    """
    主函数：读取数据，计算标准化后的整体趋势，为每个类别生成图。
    """
    # 1. 加载数据
    means_df_raw, class_series = load_and_group_data(file_path)

    # 2. 标准化（每个代谢物跨6个处理组Z-score）
    means_df_std = standardize_across_groups(means_df_raw)

    # 3. 获取所有类别
    categories = class_series.unique()
    periods = ['G', 'R']
    genotypes = ['lht1', 'WT', 'OX']

    # 4. 存储回归结果（可选）
    results = []

    for cat in categories:
        print(f"处理类别: {cat}")
        trend = class_trend(cat, means_df_std, class_series, periods, genotypes)
        if trend is None:
            continue
        mean_by_period, sem_by_period = trend
        plot_class_trend(cat, mean_by_period, sem_by_period, output_dir,
                         periods, genotypes)

        # 可选：记录回归结果
        for period in periods:
            doses = [-1, 0, 1]
            means = [mean_by_period[period][g] for g in genotypes]
            slope, intercept, r_value, p_value, _ = stats.linregress(doses, means)
            results.append({
                'class': cat,
                'period': period,
                'slope': slope,
                'R_squared': r_value ** 2,
                'p_value': p_value
            })

    # 保存汇总表格
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(os.path.join(output_dir, "linear_trend_summary.csv"), index=False)
        print(f"回归汇总表已保存至 {output_dir}/linear_trend_summary.csv")

    print("建模完成！")

# ------------------------------
# 6. 运行示例
# ------------------------------
if __name__ == "__main__":
    # 请将 'all_meta_values.xlsx' 替换为实际文件路径
    data_file = "all_meta_values.xlsx"
    build_trend_models(data_file, output_dir="class_trend_results")