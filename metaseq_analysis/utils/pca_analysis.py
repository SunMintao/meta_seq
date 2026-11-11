import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

class PCAAnalyzer:
    """
    对meta数据进行 PCA 分析，生成可视化图形和输出文件。
    支持按样本名前缀自动分组，并在二维 PCA 图中绘制 95% 置信椭圆。
    输入 Excel 文件：行 = meta，列 = 样本（第一列为meta ID）。
    """

    def __init__(self, file_path, sheet_name=0, index_col=0):
        """
        参数:
            file_path (str): Excel 文件路径
            sheet_name (str/int): 工作表名或索引
            index_col (int): 用作行索引的列号（meta ID 列）
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.index_col = index_col
        self.data = None
        self.X = None
        self.sample_names = None
        self.gene_names = None
        self.pca = None
        self.scores = None        # 样本主成分得分 (n_samples, n_components)
        self.loadings = None
        self.explained_variance_ratio = None
        self.components = None
        self.groups = None         # 每个样本对应的分组标签

    def load_data(self):
        """读取 Excel，转置为样本行、meta列，并提取样本/meta名及分组信息"""
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=self.index_col)
        self.data = df.copy()
        df_T = df.T
        self.sample_names = df_T.index.tolist()
        self.gene_names = df_T.columns.tolist()
        self.X_raw = df_T.values.astype(float)

        # 自动提取分组：去掉末尾的复制编号（如 -1, -2, -3）
        self._extract_groups()
        print(f"数据加载完成：{self.X_raw.shape[0]} 个样本，{self.X_raw.shape[1]} 个meta")
        print(f"发现分组：{set(self.groups)}")

    def _extract_groups(self):
        """根据样本名自动分组，规则：移除最后一个 '-' 及其后面的数字部分"""
        groups = []
        for name in self.sample_names:
            parts = name.split('-')
            if len(parts) > 1 and parts[-1].isdigit():
                group = '-'.join(parts[:-1])
            else:
                group = name   # 保底方案
            groups.append(group)
        self.groups = groups

    def standardize(self):
        """对每个meta（特征）进行标准化（均值为0，方差为1）"""
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X_raw)
        print("数据标准化完成")

    def perform_pca(self, n_components=None):
        """
        执行 PCA
        参数:
            n_components (int): 保留的主成分个数，默认为 min(样本数, 特征数)
        """
        if self.X is None:
            raise ValueError("请先调用 load_data() 和 standardize() 加载并标准化数据")
        if n_components is None:
            n_components = min(self.X.shape)
        self.pca = PCA(n_components=n_components)
        self.scores = self.pca.fit_transform(self.X)
        self.components = self.pca.components_
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.loadings = pd.DataFrame(
            self.components.T,
            index=self.gene_names,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        print(f"PCA 完成，保留 {n_components} 个主成分，累计方差解释率: {self.explained_variance_ratio.sum():.2%}")

    def _confidence_ellipse(self, x, y, ax, n_std=2.447, **kwargs):
        """
        计算并绘制置信椭圆（基于协方差矩阵）
        参数:
            x, y: 数据点坐标
            ax: matplotlib axes 对象
            n_std: 椭圆半径倍数，对于 95% 置信区间（2自由度），n_std = sqrt(chi2(0.95,2)) = 2.447
            **kwargs: 传递给 Ellipse 的参数
        """
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)
        # 计算椭圆特征值和特征向量
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                          angle=theta, **kwargs)
        ax.add_patch(ellipse)

    def plot_2d(self, save_path=None, figsize=(10, 8), conf_level=0.95):
        """绘制二维 PCA 散点图（PC1 vs PC2），按分组着色并绘制置信椭圆"""
        if self.scores is None:
            raise RuntimeError("请先执行 perform_pca()")
        if self.scores.shape[1] < 2:
            print("主成分不足2个，无法绘制二维图")
            return

        plt.figure(figsize=figsize)
        ax = plt.gca()
        unique_groups = np.unique(self.groups)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

        # 计算卡方阈值（95% 置信，自由度=2）
        from scipy.stats import chi2
        chi2_val = chi2.ppf(conf_level, df=2)
        n_std = np.sqrt(chi2_val)   # 2.447 for 0.95

        # 分别绘制每组数据，并画椭圆
        for group, color in zip(unique_groups, colors):
            idx = [i for i, g in enumerate(self.groups) if g == group]
            if not idx:
                continue
            pc1_vals = self.scores[idx, 0]
            pc2_vals = self.scores[idx, 1]
            plt.scatter(pc1_vals, pc2_vals, label=group, color=color, s=60, alpha=0.8)
            # 绘制置信椭圆（仅当该组样本数 >= 3 时，否则需协方差矩阵奇异）
            if len(idx) >= 3:
                self._confidence_ellipse(pc1_vals, pc2_vals, ax, n_std=n_std,
                                         edgecolor=color, facecolor='none', linewidth=1.5, linestyle='--')
            # 可选：标记组中心
            plt.scatter(np.mean(pc1_vals), np.mean(pc2_vals), marker='X', s=100, color=color, edgecolors='black')

        plt.xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%})')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%})')
        plt.title('PCA 2D Projection with 95% Confidence Ellipses')
        plt.legend(title='Group')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"二维图已保存至 {save_path}")
        else:
            plt.show()

    def plot_3d(self, save_path=None, figsize=(10, 8)):
        """绘制三维 PCA 散点图（前三个主成分），按分组着色（无置信椭球）"""
        if self.scores is None:
            raise RuntimeError("请先执行 perform_pca()")
        if self.scores.shape[1] < 3:
            print("警告：主成分数不足3，无法绘制三维图")
            return

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        unique_groups = np.unique(self.groups)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

        for group, color in zip(unique_groups, colors):
            idx = [i for i, g in enumerate(self.groups) if g == group]
            if not idx:
                continue
            xs = self.scores[idx, 0]
            ys = self.scores[idx, 1]
            zs = self.scores[idx, 2]
            ax.scatter(xs, ys, zs, label=group, color=color, s=50, alpha=0.8)
            # 标记组中心
            ax.scatter(np.mean(xs), np.mean(ys), np.mean(zs), marker='X', s=100, color=color, edgecolors='black')

        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%})')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%})')
        ax.set_zlabel(f'PC3 ({self.explained_variance_ratio[2]:.2%})')
        ax.set_title('PCA 3D Projection')
        ax.legend(title='Group')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"三维图已保存至 {save_path}")
        else:
            plt.show()

    def save_loadings(self, output_path="loadings.csv"):
        """保存载荷矩阵（meta × 主成分）到 CSV"""
        if self.loadings is None:
            raise RuntimeError("请先执行 perform_pca()")
        self.loadings.to_csv(output_path, index=True)
        print(f"载荷矩阵已保存至 {output_path}")

    def save_scores(self, output_path="factor_scores.csv"):
        """保存每个样本的主成分得分（因子得分）到 CSV"""
        if self.scores is None:
            raise RuntimeError("请先执行 perform_pca()")
        scores_df = pd.DataFrame(
            self.scores,
            index=self.sample_names,
            columns=[f"PC{i+1}" for i in range(self.scores.shape[1])]
        )
        scores_df.to_csv(output_path, index=True)
        print(f"样本主成分得分表已保存至 {output_path}")

    def save_factor_analysis_result(self, output_path="factor_analysis_result.txt"):
        """输出因子分析结果：特征值、方差解释率、累计解释率、载荷摘要"""
        if self.pca is None:
            raise RuntimeError("请先执行 perform_pca()")
        eigenvalues = self.pca.explained_variance_
        ev_ratio = self.explained_variance_ratio
        cum_ratio = np.cumsum(ev_ratio)
        n_comp = len(eigenvalues)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("========== PCA 因子分析结果 ==========\n\n")
            f.write("特征值 (Eigenvalues):\n")
            for i, val in enumerate(eigenvalues, 1):
                f.write(f"PC{i}: {val:.4f}\n")
            f.write("\n方差解释比例 (Explained Variance Ratio):\n")
            for i, val in enumerate(ev_ratio, 1):
                f.write(f"PC{i}: {val:.4f} ({val:.2%})\n")
            f.write("\n累计方差解释比例:\n")
            for i, val in enumerate(cum_ratio, 1):
                f.write(f"前{i}个主成分: {val:.4f} ({val:.2%})\n")
            f.write("\n载荷矩阵 (Loadings，前5个主成分，仅展示前10个meta):\n")
            f.write(self.loadings.iloc[:10, :min(5, n_comp)].to_string())
            f.write("\n\n完整载荷矩阵请参见 loadings.csv 文件")
        print(f"因子分析结果已保存至 {output_path}")

    def run_full_analysis(self, n_components=None, output_dir="."):
        """执行完整的分析流程，生成所有输出"""
        import os
        self.load_data()
        self.standardize()
        self.perform_pca(n_components=n_components)
        self.plot_2d(save_path=os.path.join(output_dir, "pca_2d.png"))
        if self.scores.shape[1] >= 3:
            self.plot_3d(save_path=os.path.join(output_dir, "pca_3d.png"))
        self.save_loadings(os.path.join(output_dir, "loadings.csv"))
        self.save_scores(os.path.join(output_dir, "factor_scores.csv"))
        self.save_factor_analysis_result(os.path.join(output_dir, "factor_analysis_result.txt"))
        print("\n所有分析完成！")


if __name__ == "__main__":
    # 使用示例（请确保 "1.xlsx" 在当前目录下）
    analyzer = PCAAnalyzer("1.xlsx", sheet_name="Sheet1", index_col=0)
    analyzer.run_full_analysis(output_dir=".")