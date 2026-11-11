import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2

def plsda_scores_plot(data_file, group_file, group_id, n_components=2, scale=True,
                      show_labels=True, show_conf_int=True, conf_level=0.95,
                      save_path=None, figsize=(8,6), dpi=300):
    """
    执行 PLS-DA 分析并绘制前两个成分的得分图（可选置信椭圆）。

    参数
    ----------
    data_file : str
        代谢物数据 Excel 文件路径，必须包含 'Index', 'Compounds' 及样本列。
    group_file : str
        分组信息 Excel 文件路径，包含 'samples', 'condition', 'group' 三列。
    group_id : int or str
        要分析的分组编号（对应 group 列中的值）。
    n_components : int, default=2
        PLS 成分个数（将取前两个绘图）。
    scale : bool, default=True
        是否对代谢物数据进行标准化（均值为0，方差为1）。
    show_labels : bool, default=True
        是否在散点上显示样本名称。
    show_conf_int : bool, default=True
        是否在图上添加置信椭圆（每组一个）。
    conf_level : float, default=0.95
        置信水平（如0.95表示95%置信区间）。
    save_path : str or None, default=None
        若指定路径，则保存图片；否则调用 plt.show() 显示。
    figsize : tuple, default=(8,6)
        图形尺寸（宽, 高）。
    dpi : int, default=300
        保存图片时的分辨率。

    返回
    -------
    fig : matplotlib.figure.Figure
        绘制的图形对象。
    scores : np.ndarray
        样本得分矩阵，形状 (n_samples, n_components)。
    model : PLSRegression
        拟合的 PLS 模型对象。
    """
    # ---------- 1. 读取数据 ----------
    data_df = pd.read_excel(data_file)
    group_df = pd.read_excel(group_file)

    required_data = ['Index', 'Compounds']
    for col in required_data:
        if col not in data_df.columns:
            raise KeyError(f"数据文件缺少必需列: {col}")
    required_group = ['samples', 'condition', 'group']
    for col in required_group:
        if col not in group_df.columns:
            raise KeyError(f"分组文件缺少必需列: {col}")

    # ---------- 2. 提取指定分组 ----------
    sub_group = group_df[group_df['group'] == group_id].copy()
    if sub_group.empty:
        raise ValueError(f"分组文件中没有 group = {group_id} 的记录")

    samples_condition = dict(zip(sub_group['samples'], sub_group['condition']))
    sample_names = list(samples_condition.keys())
    conditions = list(samples_condition.values())

    data_sample_cols = [c for c in data_df.columns if c not in required_data]
    missing_samples = set(sample_names) - set(data_sample_cols)
    if missing_samples:
        raise ValueError(f"以下样本在数据文件中不存在: {missing_samples}")

    # ---------- 3. 构建 X 和 Y ----------
    X_raw = data_df[sample_names].T.values
    le = LabelEncoder()
    Y = le.fit_transform(conditions).reshape(-1, 1)

    # ---------- 4. 数据标准化（可选） ----------
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    # ---------- 5. 调整成分数 ----------
    n_samples, n_features = X.shape
    n_classes = len(le.classes_)
    max_comp = min(n_samples - 1, n_features, n_classes - 1)
    if n_components > max_comp:
        print(f"警告: n_components={n_components} 超过上限 {max_comp}，已调整为 {max_comp}")
        n_components = max_comp
    if n_components < 1:
        raise RuntimeError("无法建立 PLS 模型，请检查样本数、特征数或类别数")

    # ---------- 6. 拟合 PLS-DA 模型 ----------
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    scores = pls.x_scores_
    if scores.shape[1] >= 2:
        score_x = scores[:, 0]
        score_y = scores[:, 1]
    else:
        score_x = scores[:, 0]
        score_y = np.zeros_like(score_x)
        print("警告: 成分数不足2，第二维已填充为0")

    # ---------- 7. 计算各成分的 R²Y ----------
    T = pls.x_scores_
    R2Y = []
    for h in range(T.shape[1]):
        t = T[:, h]
        corr = np.corrcoef(t, Y.flatten())[0, 1]
        R2Y.append(corr ** 2)
    r2y_comp1 = R2Y[0] if len(R2Y) > 0 else 0
    r2y_comp2 = R2Y[1] if len(R2Y) > 1 else 0

    # ---------- 8. 绘图 ----------
    fig, ax = plt.subplots(figsize=figsize)

    unique_conds = np.unique(conditions)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_conds)))
    color_map = dict(zip(unique_conds, colors))

    # 绘制散点（添加图例标签）
    for cond in unique_conds:
        idx = [i for i, c in enumerate(conditions) if c == cond]
        ax.scatter(score_x[idx], score_y[idx],
                   label=cond, c=[color_map[cond]], s=80, edgecolor='k', alpha=0.8)

    # 添加置信椭圆（不添加图例标签，因此不会出现在图例中）
    if show_conf_int and scores.shape[1] >= 2:
        chi2_val = chi2.ppf(conf_level, df=2)
        for cond in unique_conds:
            idx = np.array([i for i, c in enumerate(conditions) if c == cond])
            if len(idx) < 3:
                print(f"警告: 组 '{cond}' 样本数少于3，无法绘制可靠置信椭圆，跳过")
                continue
            group_scores = scores[idx, :2]
            mean_vec = group_scores.mean(axis=0)
            cov_mat = np.cov(group_scores, rowvar=False)
            eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
            width = 2 * np.sqrt(chi2_val * eig_vals[1])
            height = 2 * np.sqrt(chi2_val * eig_vals[0])
            angle = np.degrees(np.arctan2(eig_vecs[1, 1], eig_vecs[0, 1]))
            ellipse = Ellipse(xy=mean_vec, width=width, height=height,
                              angle=angle, edgecolor=color_map[cond],
                              facecolor='none', linewidth=2, linestyle='--')
                              # 没有 label 参数 → 椭圆不出现在图例中
            ax.add_patch(ellipse)

    if show_labels:
        for i, name in enumerate(sample_names):
            ax.annotate(name, (score_x[i], score_y[i]),
                        textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

    ax.set_xlabel(f"PLS Component 1 (R²Y = {r2y_comp1:.2f})", fontsize=12)
    ax.set_ylabel(f"PLS Component 2 (R²Y = {r2y_comp2:.2f})", fontsize=12)
    ax.set_title(f"PLS-DA Scores Plot (Group {group_id})", fontsize=14)

    # 图例放在图的内部（右上角），只显示散点的三个类别
    ax.legend(title="Condition", loc='best', frameon=True, fancybox=True)

    ax.grid(alpha=0.3)
    fig.tight_layout()   # 自动调整布局，图例在内部不会出界

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

    return fig, scores, pls