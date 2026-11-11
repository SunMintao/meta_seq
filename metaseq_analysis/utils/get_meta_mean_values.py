import pandas as pd
import re


def compute_group_means(file_path):
    """
    读取Excel文件，计算每个化合物在不同实验组（如G-lht1、G-WT等）中的均值。

    参数:
        file_path (str): Excel文件路径

    返回:
        pd.DataFrame: 包含'Compounds'列和每组均值列（如'G-lht1_mean'）的DataFrame
    """
    # 读取Excel数据
    df = pd.read_excel(file_path, sheet_name=0)

    # 获取所有列名
    cols = df.columns.tolist()

    # 识别样本列（所有以'-数字'结尾的列）
    sample_cols = []
    group_map = {}  # 用于存储每个样本列对应的组名
    pattern = re.compile(r'-\d+$')  # 匹配末尾的-数字

    for col in cols:
        if pattern.search(str(col)):
            sample_cols.append(col)
            # 去掉末尾的-数字作为组名
            group_name = re.sub(r'-\d+$', '', col)
            group_map[col] = group_name

    # 如果没有找到样本列，则返回空DataFrame
    if not sample_cols:
        print("未找到样本列（以-数字结尾的列名）")
        return pd.DataFrame()

    # 获取所有唯一的组名
    unique_groups = sorted(set(group_map.values()))

    # 初始化结果DataFrame，包含'Compounds'列
    result = df[['Compounds']].copy()

    # 对每个组，计算对应样本列的均值
    for group in unique_groups:
        # 找出属于该组的所有样本列
        group_cols = [col for col, g in group_map.items() if g == group]
        # 计算每行的均值（忽略NaN）
        mean_col_name = f"{group}"
        result[mean_col_name] = df[group_cols].mean(axis=1, skipna=True)

    return result


# 使用示例
if __name__ == "__main__":
    file_path = "all_meta_values.xlsx"
    result_df = compute_group_means(file_path)
    print(result_df.head())