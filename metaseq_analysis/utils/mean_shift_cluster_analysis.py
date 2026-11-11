import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns


class MetaContentCluster:
    """
    基因表达聚类分析类
    流程：读取数据 -> 行标准化（Z-score） -> Mean-Shift聚类 ->
          可选合并至指定最大簇数 -> 保存结果 -> 绘制趋势图与PCA图
    """

    def __init__(self, file_path, sheet_name=0, sample_order=None, max_clusters=10):
        """
        初始化
        Parameters:
        -----------
        file_path : str
            Excel 文件路径
        sheet_name : str or int
            Excel 工作表名或索引
        sample_order : list of str, optional
            期望的样本列顺序，若为 None 则使用文件中的原始顺序
        max_clusters : int
            最终允许的最大簇数（若初始簇数超过该值则合并）
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.sample_order = sample_order
        self.max_clusters = max_clusters

        self.df_raw = None  # 原始数据（基因 x 样本）
        self.X_raw = None  # 原始表达矩阵
        self.X_scaled = None  # 行标准化后的矩阵
        self.scaler = None  # StandardScaler 对象
        self.labels = None  # 最终聚类标签
        self.cluster_centers = None  # 最终聚类中心（标准化空间中）
        self.n_clusters = None  # 最终簇数

    def load_data(self):
        """读取 Excel 文件，并按指定样本顺序排列"""
        self.df_raw = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=0)
        if self.sample_order is not None:
            # 确保列都存在
            missing = set(self.sample_order) - set(self.df_raw.columns)
            if missing:
                raise ValueError(f"样本列缺失: {missing}")
            self.df_raw = self.df_raw[self.sample_order]
        self.X_raw = self.df_raw.values.astype(float)
        print(f"数据加载完成: {self.X_raw.shape[0]} 个基因, {self.X_raw.shape[1]} 个样本")
        return self

    def standardize(self):
        """按基因（行）进行 Z-score 标准化"""
        self.scaler = StandardScaler()
        # 转置使每列为一个基因，标准化后转置回来
        self.X_scaled = self.scaler.fit_transform(self.X_raw.T).T
        print("行标准化 (Z-score) 完成")
        print(f"标准化后的范围: min={self.X_scaled.min():.3f}, max={self.X_scaled.max():.3f}")
        # 验证第一个基因的均值和标准差
        print(f"第一个基因: mean={self.X_scaled[0].mean():.6f}, std={self.X_scaled[0].std():.6f}")
        return self

    def mean_shift_cluster(self, quantile=0.2, n_samples=500, bin_seeding=True):
        """
        执行 Mean-Shift 聚类
        Parameters:
        -----------
        quantile : float
            带宽估计的分位数，值越小聚类越多
        n_samples : int
            估计带宽时使用的样本数
        bin_seeding : bool
            是否使用 bin seeding 加速
        """
        print("估计带宽...")
        bandwidth = estimate_bandwidth(self.X_scaled, quantile=quantile, n_samples=n_samples)
        print(f"带宽 = {bandwidth:.4f}")

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        ms.fit(self.X_scaled)
        initial_labels = ms.labels_
        initial_centers = ms.cluster_centers_
        n_initial = len(np.unique(initial_labels))
        print(f"Mean-Shift 初始簇数: {n_initial}")

        # 若初始簇数超过最大允许值，进行合并
        if n_initial > self.max_clusters:
            print(f"初始簇数 {n_initial} > {self.max_clusters}，开始合并...")
            self._merge_clusters(initial_labels, initial_centers)
        else:
            self.labels = initial_labels
            self.cluster_centers = initial_centers
            self.n_clusters = n_initial
            print(f"最终簇数: {self.n_clusters}")
        return self

    def _merge_clusters(self, initial_labels, initial_centers):
        """
        基于簇中心的层次聚类合并簇至 self.max_clusters
        """
        # 计算各簇中心之间的距离并做层次聚类
        Z = linkage(initial_centers, method='ward')
        # 直接截断到 max_clusters 个簇
        merged = fcluster(Z, t=self.max_clusters, criterion='maxclust')  # 返回 1 ~ max_clusters
        # 建立原簇id -> 新簇id 的映射 (0-indexed)
        mapping = {old: new - 1 for old, new in enumerate(merged)}
        self.labels = np.array([mapping[l] for l in initial_labels])

        # 重新计算每个新簇的中心（标准化空间中的均值）
        new_centers = []
        for k in range(self.max_clusters):
            mask = (self.labels == k)
            if np.any(mask):
                new_centers.append(self.X_scaled[mask].mean(axis=0))
            else:
                new_centers.append(np.zeros(self.X_scaled.shape[1]))
        new_centers = np.array(new_centers)

        # 移除实际可能存在的空簇（理论上 fcluster 不会产生空簇，但安全处理）
        present = np.unique(self.labels)
        if len(present) < self.max_clusters:
            relabel_map = {old: i for i, old in enumerate(sorted(present))}
            self.labels = np.array([relabel_map[l] for l in self.labels])
            new_centers = new_centers[list(present)]
            self.n_clusters = len(present)
        else:
            self.n_clusters = self.max_clusters
        self.cluster_centers = new_centers
        print(f"合并后簇数: {self.n_clusters}")

    def save_results(self, output_excel="cluster_results.xlsx"):
        """保存带簇标签的原始数据到 Excel"""
        if self.df_raw is None or self.labels is None:
            raise RuntimeError("请先运行 load_data() 和 mean_shift_cluster()")
        df_out = self.df_raw.copy()
        df_out['Cluster'] = self.labels
        df_out.to_excel(output_excel, index=True)
        print(f"聚类结果已保存至: {output_excel}")
        return output_excel

    def plot_trends(self, output_image="cluster_trends.png"):
        """
        绘制每个簇的标准化均值趋势图（含 95% 置信区间）
        """
        if self.X_scaled is None or self.labels is None:
            raise RuntimeError("请先完成聚类")

        # 构建带标签的 DataFrame（标准化值）
        df_scaled = pd.DataFrame(self.X_scaled, index=self.df_raw.index, columns=self.df_raw.columns)
        df_scaled['Cluster'] = self.labels

        conditions = self.df_raw.columns.tolist()
        n_clusters = self.n_clusters
        n_cols = min(4, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_clusters == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for k in range(n_clusters):
            cluster_metas = df_scaled[df_scaled['Cluster'] == k].index
            if len(cluster_metas) == 0:
                continue
            cluster_expr = df_scaled.loc[cluster_metas, conditions]
            means = cluster_expr.mean(axis=0).values
            if len(cluster_metas) > 1:
                sem = cluster_expr.sem(axis=0).values
                t_val = stats.t.ppf(0.975, len(cluster_metas) - 1)
                ci_lower = means - t_val * sem
                ci_upper = means + t_val * sem
            else:
                ci_lower = ci_upper = means

            ax = axes[k]
            x = np.arange(len(conditions))
            ax.plot(x, means, marker='o', label=f'Cluster {k} (n={len(cluster_metas)})')
            ax.fill_between(x, ci_lower, ci_upper, alpha=0.2)
            ax.set_xticks(x)
            ax.set_xticklabels(conditions, rotation=45, ha='right')
            ax.set_ylabel('Mean Z-score')
            ax.set_title(f'Cluster {k} (n={len(cluster_metas)})')
            ax.grid(alpha=0.3)
            ax.legend()

        # 隐藏多余子图
        for k in range(n_clusters, len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        plt.show()
        print(f"趋势图已保存至: {output_image}")
        return output_image

    def plot_pca(self, output_image="pca_clusters.png"):
        """
        绘制 PCA 降维后的散点图，并用椭圆表示 95% 置信区间
        """
        if self.X_scaled is None or self.labels is None:
            raise RuntimeError("请先完成聚类")

        # PCA 降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        expl_var = pca.explained_variance_ratio_
        print(f"PCA 解释方差: PC1={expl_var[0]:.3f}, PC2={expl_var[1]:.3f}")

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("tab10", self.n_clusters)

        for k in range(self.n_clusters):
            mask = (self.labels == k)
            if np.sum(mask) == 0:
                continue
            points = X_pca[mask]
            plt.scatter(points[:, 0], points[:, 1], c=[colors[k]], label=f'Cluster {k}', alpha=0.7, s=50)

            # 绘制置信椭圆（若点数 > 2）
            if len(points) > 2:
                mean_vec = np.mean(points, axis=0)
                cov = np.cov(points.T)
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
                width, height = 2 * np.sqrt(eig_vals) * 2  # 2倍标准差覆盖约95%点
                ellipse = Ellipse(xy=mean_vec, width=width, height=height, angle=angle,
                                  edgecolor=colors[k], facecolor='none', linewidth=2, linestyle='--')
                plt.gca().add_patch(ellipse)

        # 绘制簇中心（PCA空间）
        if self.cluster_centers is not None:
            centers_pca = pca.transform(self.cluster_centers)
            plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, edgecolors='white',
                        linewidth=1.5)

        plt.xlabel(f'PC1 ({expl_var[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({expl_var[1] * 100:.1f}%)')
        plt.title(f'Mean-Shift cluster analysis')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        plt.show()
        print(f"PCA 图已保存至: {output_image}")
        return output_image

    def run_pipeline(self, trend_image="trends.png", pca_image="pca.png", result_excel="results.xlsx"):
        """
        执行完整流程：加载 -> 标准化 -> 聚类 -> 保存结果 -> 绘制趋势图 -> 绘制PCA图
        """
        self.load_data()
        self.standardize()
        self.mean_shift_cluster()
        self.save_results(result_excel)
        self.plot_trends(trend_image)
        self.plot_pca(pca_image)
        print("\n=== 分析完成 ===")
        return self


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 实例化对象，指定文件路径和样本顺序
    analyzer = MetaContentCluster(
        file_path="all_diff_meta_ordered_column_tpm_mean.xlsx",
        sheet_name="Sheet1",
        sample_order=['L-lht1', 'L-WT', 'L-OX', 'H-lht1', 'H-WT', 'H-OX'],
        max_clusters=10
    )

    # 运行完整流程
    analyzer.run_pipeline(
        trend_image="cluster_trends_scaled_CI_corrected.png",
        pca_image="clustering_pca_merged_corrected.png",
        result_excel="mean_shift_cluster_results_corrected.xlsx"
    )