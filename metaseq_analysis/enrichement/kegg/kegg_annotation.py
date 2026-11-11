import pandas as pd
import requests
import time
from urllib.parse import quote
from typing import Optional, Tuple

# KEGG API 基础 URL
KEGG_FIND = "http://rest.kegg.jp/find/compound"
KEGG_GET = "http://rest.kegg.jp/get"

def search_kegg_compound(compound_name: str) -> Optional[str]:
    """通过化合物名称搜索 KEGG，返回第一个匹配的化合物 ID (Cxxxxx)"""
    if not compound_name or pd.isna(compound_name):
        return None
    try:
        encoded_name = quote(str(compound_name))
        url = f"{KEGG_FIND}/{encoded_name}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split('\n')
        for line in lines:
            if line.startswith('cpd:'):
                cpd_id = line.split('\t')[0].split(':')[1]
                return cpd_id
    except Exception as e:
        print(f"搜索化合物 {compound_name} 失败: {e}")
    return None

def get_compound_reactions(cpd_id: str) -> list:
    """获取 KEGG 化合物参与的所有反应 ID (Rxxxxx)"""
    if not cpd_id:
        return []
    try:
        url = f"{KEGG_GET}/{cpd_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        content = resp.text
        reactions = []
        for line in content.split('\n'):
            if line.startswith('REACTION'):
                parts = line.split()
                for part in parts[1:]:
                    if part.startswith('R'):
                        reactions.append(part)
        return reactions
    except Exception as e:
        print(f"获取化合物 {cpd_id} 的反应失败: {e}")
        return []

def get_reaction_kos(reaction_id: str) -> list:
    """根据反应 ID 获取该反应关联的 KO 编号列表"""
    if not reaction_id:
        return []
    try:
        url = f"{KEGG_GET}/{reaction_id}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        content = resp.text
        kos = []
        for line in content.split('\n'):
            if line.startswith('ORTHOLOGY'):
                parts = line.split()
                for part in parts[1:]:
                    if part.startswith('K'):
                        kos.append(part)
        return kos
    except Exception as e:
        print(f"获取反应 {reaction_id} 的 KO 失败: {e}")
        return []

def get_kegg_annotation(compound_name: str, delay: float = 0.2) -> Tuple[Optional[str], str]:
    """
    对单个代谢物进行 KEGG 注释
    返回 (cpd_id, ko_string)
    """
    cpd_id = search_kegg_compound(compound_name)
    time.sleep(delay)
    if not cpd_id:
        return None, ""

    reactions = get_compound_reactions(cpd_id)
    all_kos = set()
    for rid in reactions:
        kos = get_reaction_kos(rid)
        all_kos.update(kos)
        time.sleep(delay)
    ko_string = ";".join(sorted(all_kos)) if all_kos else ""
    return cpd_id, ko_string

def annotate_metabolites(
    input_file: str,
    output_file: Optional[str] = None,
    sheet_name: str = 0,
    compound_col: str = "Compounds",
    output_cpd_col: str = "KEGG_Compound_ID",
    output_ko_col: str = "KEGG_KO",
    delay: float = 0.5,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    对代谢物文件进行 KEGG 注释（仅使用 Compounds 列），返回注释后的 DataFrame。

    参数:
        input_file: 输入 Excel 文件路径
        output_file: 输出 Excel 文件路径，若为 None 则不保存
        sheet_name: 工作表名或索引，默认为 0
        compound_col: 代谢物名称所在列名，默认为 "Compounds"
        output_cpd_col: 输出的 KEGG 化合物 ID 列名，默认为 "KEGG_Compound_ID"
        output_ko_col: 输出的 KO 编号列名，默认为 "KEGG_KO"
        delay: 每次 API 请求间隔（秒），避免过快
        skip_existing: 是否跳过已注释的行（基于 output_cpd_col 非空）
    返回:
        注释后的 DataFrame
    """
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    if output_cpd_col not in df.columns:
        df[output_cpd_col] = ""
    if output_ko_col not in df.columns:
        df[output_ko_col] = ""

    total = len(df)
    for idx, row in df.iterrows():
        if skip_existing and pd.notna(row[output_cpd_col]) and str(row[output_cpd_col]).strip() != "":
            continue
        compound_name = row[compound_col]
        print(f"处理 {idx+1}/{total}: {compound_name}")
        cpd_id, ko_str = get_kegg_annotation(compound_name, delay=delay)
        df.at[idx, output_cpd_col] = cpd_id if cpd_id else ""
        df.at[idx, output_ko_col] = ko_str
        time.sleep(delay)

    if output_file:
        df.to_excel(output_file, index=False)
        print(f"结果已保存至: {output_file}")
    return df

if __name__ == "__main__":
    # 示例：直接运行
    result = annotate_metabolites(
        input_file="all_meta_annotation.xlsx",
        output_file="all_meta_annotation_KEGG.xlsx",
        compound_col="Compounds"
    )
    print("注释完成，前5行:")
    print(result.head())