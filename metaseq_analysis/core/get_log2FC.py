import numpy as np
import pandas as pd


def compute_log2fc(data_file, group_file, pseudo_count=1e-10, fc_direction='b_vs_a'):
    """
    根据样本分组文件，计算每个代谢物的 log2FC。

    参数
    ----------
    data_file : str
        代谢物数据文件路径（Excel），应包含 Index、Compounds 及各样本列。
    group_file : str
        分组文件路径（Excel），包含三列：samples（样本名）、condition（a/b）、group（分组编号）。
    pseudo_count : float, default=1e-10
        添加到均值的伪计数，避免 log(0) 或除零。
    fc_direction : str, default='b_vs_a'
        比较方向，'b_vs_a' 表示 log2(b/a)，'a_vs_b' 表示 log2(a/b)。

    返回
    -------
    pd.DataFrame
        包含 Index、Compounds 以及每一组的 log2FC（列名为 log2FC_group_<组号>）的 DataFrame。
    """
    # 读取数据
    data_df = pd.read_excel(data_file)
    group_df = pd.read_excel(group_file)

    # 检查必需列
    if 'Index' not in data_df.columns or 'Compounds' not in data_df.columns:
        raise KeyError("data_file 中必须包含 'Index' 和 'Compounds' 列")
    required_group_cols = ['samples', 'condition', 'group']
    for col in required_group_cols:
        if col not in group_df.columns:
            raise KeyError(f"group_file 中必须包含 '{col}' 列")

    # 获取所有样本列（即 data_df 中除了 Index 和 Compounds 以外的列）
    sample_cols = [c for c in data_df.columns if c not in ['Index', 'Compounds']]

    # 确保样本列存在于 group_df 的 samples 中
    group_samples = set(group_df['samples'])
    data_samples = set(sample_cols)
    missing_in_group = data_samples - group_samples
    if missing_in_group:
        print(f"警告：数据中的样本 {missing_in_group} 不在分组文件中，将被忽略")

    # 仅保留同时出现在数据与分组文件中的样本
    valid_samples = list(group_samples & data_samples)
    if not valid_samples:
        raise ValueError("数据和分组文件中没有共同的样本名")

    # 按 group 分组，每个组内分别提取 condition a 和 b 的样本列表
    groups = group_df.groupby('group')
    log2fc_results = {'Index': data_df['Index'], 'Compounds': data_df['Compounds']}

    for group_id, sub_df in groups:
        # 获取该组内 condition 为 a 和 b 的样本列表
        samples_a = sub_df[sub_df['condition'] == 'a']['samples'].tolist()
        samples_b = sub_df[sub_df['condition'] == 'b']['samples'].tolist()

        if not samples_a or not samples_b:
            print(f"警告：group {group_id} 中缺少 condition a 或 b，跳过该组")
            continue

        # 确保样本存在于数据中
        samples_a = [s for s in samples_a if s in data_df.columns]
        samples_b = [s for s in samples_b if s in data_df.columns]

        if not samples_a or not samples_b:
            print(f"警告：group {group_id} 中有效样本不足，跳过")
            continue

        # 提取数据
        data_a = data_df[samples_a].values  # shape: n_metabolites x n_samples_a
        data_b = data_df[samples_b].values  # shape: n_metabolites x n_samples_b

        # 计算均值（沿列方向，即每个代谢物的均值）
        mean_a = np.mean(data_a, axis=1)
        mean_b = np.mean(data_b, axis=1)

        # 添加伪计数
        mean_a += pseudo_count
        mean_b += pseudo_count

        # 计算 log2FC
        if fc_direction == 'b_vs_a':
            fc = mean_b / mean_a
        elif fc_direction == 'a_vs_b':
            fc = mean_a / mean_b
        else:
            raise ValueError("fc_direction 必须是 'b_vs_a' 或 'a_vs_b'")

        log2fc = np.log2(fc)

        # 存储结果
        log2fc_results[f'log2FC_group_{group_id}'] = log2fc

    # 构建结果 DataFrame
    result_df = pd.DataFrame(log2fc_results)
    return result_df
