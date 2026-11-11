import pandas as pd
from glob import glob
import os


def merge_excel_files(file_paths, key_column='Compounds', how='outer',
                      drop_index=False, save_path=None):
    """
    批量合并多个 Excel 文件，按关键列进行横向并集合并。

    参数
    ----------
    file_paths : list or str
        文件路径列表，或包含通配符的字符串（如 '*.xlsx'）
    key_column : str, default='Compounds'
        用于合并的关键列名，所有文件中必须存在该列
    how : str, default='outer'
        合并方式，'outer' 为并集（保留所有键），'inner' 为交集
    drop_index : bool, default=False
        是否删除每个文件中的 'Index' 列。
        - True: 删除 Index 列（推荐，避免冗余）
        - False: 保留 Index 列，并自动重命名冲突的列（添加后缀）
    save_path : str or None, default=None
        若指定路径，则保存合并结果到 Excel

    返回
    -------
    pd.DataFrame
        合并后的 DataFrame
    """
    if isinstance(file_paths, str):
        file_paths = glob(file_paths)
    if not file_paths:
        raise ValueError("未找到任何文件，请检查 file_paths 参数")

    # 用于存储处理后的 DataFrame 列表
    dfs = []
    for i, path in enumerate(file_paths):
        df = pd.read_excel(path)

        # 处理 Index 列
        if 'Index' in df.columns:
            if drop_index:
                df = df.drop(columns=['Index'])
            else:
                # 重命名 Index 列为唯一名称，避免后续合并冲突
                # 方案1: 使用文件名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(path))[0]
                # 如果文件名过长或含特殊字符，可以改用编号
                new_name = f"Index_{base_name}"
                # 但可能出现重复，加上编号更安全
                if new_name in df.columns:
                    new_name = f"Index_file{i + 1}"
                df = df.rename(columns={'Index': new_name})

        # 将关键列设为索引
        df = df.set_index(key_column)
        dfs.append(df)

    # 合并所有 DataFrame
    from functools import reduce
    df_merged = reduce(lambda left, right: left.join(right, how=how), dfs)

    # 重置索引，使 Compounds 恢复为普通列
    df_merged.reset_index(inplace=True)

    if save_path:
        df_merged.to_excel(save_path, index=False)
        print(f"合并结果已保存至: {save_path}")

    return df_merged


# ================= 使用示例 =================
if __name__ == "__main__":
    files = [
        'group_ttest_fdr_results.xlsx',
        'log2fc_result.xlsx',
        'vip_value.xlsx'
    ]

    # 不删除 Index 列，自动重命名
    merged = merge_excel_files(files, key_column='Compounds', drop_index=False)
    print(merged.columns.tolist())
    print(merged.head())

    # 若要删除 Index 列，设置 drop_index=True
    # merged = merge_excel_files(files, key_column='Compounds', drop_index=True)