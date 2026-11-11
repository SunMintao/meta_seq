import pandas as pd
import re

def extract_samples_data(input_data):
    """
    提取代谢组学数据中的 Index、Compounds 及样本列（列名以 -数字 结尾）。

    参数
    ----------
    input_data : str or pandas.DataFrame
        如果为字符串，则视为 Excel 文件路径；否则视为已读入的 DataFrame。

    返回
    -------
    pandas.DataFrame
        包含 Index、Compounds 及所有样本列的 DataFrame。
    """
    # 读取数据
    if isinstance(input_data, str):
        df = pd.read_excel(input_data, sheet_name=0)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data 必须是文件路径 (str) 或 pandas DataFrame")

    # 检查必需列
    required_cols = ['Index', 'Compounds']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"数据中缺少必需列: {col}")

    # 识别样本列：列名以 "-数字" 结尾
    sample_pattern = re.compile(r'-\d+$')
    sample_cols = [col for col in df.columns if sample_pattern.search(str(col))]

    # 提取目标列
    result_cols = ['Index', 'Compounds'] + sample_cols
    result_df = df[result_cols].copy()

    return result_df




