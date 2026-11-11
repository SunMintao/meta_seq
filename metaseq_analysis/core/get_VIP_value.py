import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

def compute_all_groups_vip(data_file, group_file, n_components=2, scale=True, pseudo_count=0):
    """
    对分组文件中所有 group 分别计算 PLS-DA 的 VIP 值，并返回合并结果。

    参数
    ----------
    data_file : str
        代谢物数据 Excel 文件路径，必须包含 'Index' 和 'Compounds' 列。
    group_file : str
        分组信息 Excel 文件路径，包含 'samples', 'condition', 'group' 三列。
    n_components : int, default=2
        PLS 成分个数（每个组独立使用该值，实际会被自动调整上限）。
    scale : bool, default=True
        是否对代谢物数据进行标准化（均值为0，方差为1）。
    pseudo_count : float, default=0
        可选，在计算 VIP 前是否对数据进行伪计数加和（一般不需要，保留为0即可）。

    返回
    -------
    pd.DataFrame
        包含 'Index', 'Compounds' 以及每组的 VIP 列（列名如 'VIP_group_1', 'VIP_group_2'）。
        如果某组无法计算 VIP（如样本不足或类别数小于2），该组 VIP 列全为 NaN。
    """
    # 读取数据
    data_df = pd.read_excel(data_file)
    group_df = pd.read_excel(group_file)

    # 检查必需列
    required_data = ['Index', 'Compounds']
    for col in required_data:
        if col not in data_df.columns:
            raise KeyError(f"数据文件缺少必需列: {col}")
    required_group = ['samples', 'condition', 'group']
    for col in required_group:
        if col not in group_df.columns:
            raise KeyError(f"分组文件缺少必需列: {col}")

    # 获取所有唯一组号
    all_groups = sorted(group_df['group'].unique())
    # 准备结果 DataFrame 的基础列
    result_df = data_df[['Index', 'Compounds']].copy()

    # 对于每个组独立计算 VIP
    for gid in all_groups:
        col_name = f'VIP_group_{gid}'
        try:
            vip = _compute_vip_for_group(data_df, group_df, gid, n_components, scale, pseudo_count)
            result_df[col_name] = vip
        except Exception as e:
            print(f"警告: 计算 group {gid} 的 VIP 失败，原因: {e}，该组 VIP 将填充为 NaN")
            result_df[col_name] = np.nan

    return result_df


def _compute_vip_for_group(data_df, group_df, group_id, n_components, scale, pseudo_count):
    """内部函数：计算单个组的 VIP 值（复用前面逻辑）"""
    # 提取该组的样本及条件
    sub_group = group_df[group_df['group'] == group_id].copy()
    if sub_group.empty:
        raise ValueError(f"分组文件中没有 group = {group_id} 的记录")

    samples_condition = dict(zip(sub_group['samples'], sub_group['condition']))
    sample_names = list(samples_condition.keys())
    conditions = list(samples_condition.values())

    # 检查样本是否存在于数据文件中
    data_sample_cols = [c for c in data_df.columns if c not in ['Index', 'Compounds']]
    missing_samples = set(sample_names) - set(data_sample_cols)
    if missing_samples:
        raise ValueError(f"样本缺失: {missing_samples}")

    # 构建 X 矩阵
    X_raw = data_df[sample_names].T.values   # (n_samples, n_metabolites)

    # 编码 Y
    le = LabelEncoder()
    Y = le.fit_transform(conditions).reshape(-1, 1)
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError("组内条件类别数少于2，无法进行 PLS-DA")

    # 标准化
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        X = X_raw

    # 调整成分数
    n_samples, n_features = X.shape
    max_comp = min(n_samples - 1, n_features, n_classes - 1)
    if n_components > max_comp:
        n_components = max_comp
    if n_components < 1:
        raise ValueError("无法建立 PLS 模型（样本数、特征数或类别数不足）")

    # PLS 模型
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)

    # 计算 VIP
    W = pls.x_weights_  # (n_features, n_components)
    T = pls.x_scores_   # (n_samples, n_components)
    R2Y = []
    for h in range(n_components):
        t = T[:, h]
        corr = np.corrcoef(t, Y.flatten())[0, 1]
        R2Y.append(corr ** 2)
    total_r2 = sum(R2Y)
    if total_r2 == 0:
        raise ValueError("所有成分的 R²Y 均为0，无法计算 VIP")
    p = n_features
    vip = np.sqrt(p * np.sum([R2Y[h] * (W[:, h] ** 2) for h in range(n_components)], axis=0) / total_r2)

    # 添加伪计数（如果需要）
    if pseudo_count != 0:
        vip = vip + pseudo_count

    return vip