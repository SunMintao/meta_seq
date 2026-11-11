import pandas as pd
from typing import Union, List


def extract_compound_info(
        file_path: str,
        compound_list: List[str],
        columns_to_extract: Union[str, List[str]]
) -> pd.DataFrame:
    """
    从Excel文件中提取指定化合物所对应的列信息。

    Parameters
    ----------
    file_path : str
        Excel文件的路径（支持.xlsx或.xls）。
    compound_list : List[str]
        需要查询的化合物名称列表，例如 ['Docosatrienoic acid', 'Glucovanillin']。
    columns_to_extract : Union[str, List[str]]
        需要提取的列名，可以是单个列名字符串，也可以是列名列表。
        例如 'Class II' 或 ['Class I', 'Class II']。

    Returns
    -------
    pd.DataFrame
        包含匹配化合物及其对应列信息的DataFrame。
        如果某些化合物未找到，仍然保留该行，对应列数据为NaN。
    """
    # 读取Excel文件，默认读取第一个sheet
    df = pd.read_excel(file_path, engine='openpyxl')

    # 确保化合物列存在（根据示例数据，列名为 'Compounds'）
    if 'Compounds' not in df.columns:
        raise KeyError("Excel文件中未找到 'Compounds' 列，请检查列名。")

    # 标准化输入：columns_to_extract 转换为列表
    if isinstance(columns_to_extract, str):
        columns_to_extract = [columns_to_extract]

    # 检查所需列是否都存在
    missing_cols = [col for col in columns_to_extract if col not in df.columns]
    if missing_cols:
        raise KeyError(f"以下列在Excel中不存在: {missing_cols}")

    # 筛选出化合物名称在 compound_list 中的行
    # 注意：假设化合物名称完全一致（包括大小写和空格）
    # 如果希望忽略大小写，可使用 df['Compounds'].str.lower().isin([c.lower() for c in compound_list])
    mask = df['Compounds'].isin(compound_list)
    result_df = df.loc[mask, ['Compounds'] + columns_to_extract].copy()

    return result_df


# 使用示例
if __name__ == "__main__":
    # 文件路径（请根据实际情况修改）
    excel_file = "data.xlsx"

    # 要查询的化合物列表
    compounds = ["Docosatrienoic acid", "Glucovanillin", "Cyclamic acid"]

    # 提取 Class II 列
    class2_info = extract_compound_info(excel_file, compounds, "Class II")
    print("提取 Class II 列的结果：")
    print(class2_info)

    # 提取 Class I 和 Class II 两列
    class1_class2_info = extract_compound_info(excel_file, compounds, ["Class I", "Class II"])
    print("\n提取 Class I 和 Class II 列的结果：")
    print(class1_class2_info)