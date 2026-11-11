import pandas as pd
import requests
import time

def get_pathways_by_compound(c_number):
    """查询单个C号对应的所有通路map号，返回逗号分隔字符串，若无则返回 '-'"""
    if not c_number or pd.isna(c_number) or str(c_number).strip() == '':
        return "-"
    cid = str(c_number).strip()
    if not cid.startswith("C"):
        cid = f"C{cid}"
    url = f"https://rest.kegg.jp/link/pathway/cpd:{cid}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        pathways = []
        for line in lines:
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                p = parts[1].split(':')[1]
                if p.startswith("map"):
                    pathways.append(p)
        return ",".join(pathways) if pathways else "-"
    except Exception as e:
        print(f"查询 {c_number} 失败: {e}")
        return "-"

def generate_kegg_annotation_from_c_number(input_path, output_path=None):
    """
    仅根据1.xlsx（列: meta, C_number）生成包含通路注释的Excel文件。
    输出列: meta, C_number, kegg_map
    """
    df1 = pd.read_excel(input_path)
    
    # 确保列存在
    if 'meta' not in df1.columns or 'C_number' not in df1.columns:
        raise ValueError("输入文件必须包含 'meta' 和 'C_number' 列")
    
    # 创建输出DataFrame，复制meta和C_number
    df_out = df1[['meta', 'C_number']].copy()
    
    # 初始化kegg_map列
    df_out['kegg_map'] = "-"
    
    # 缓存
    cache = {}
    for idx, row in df_out.iterrows():
        cid = row['C_number']
        if pd.notna(cid) and str(cid).strip() != '':
            cid_str = str(cid).strip()
            if cid_str not in cache:
                print(f"查询 {cid_str} 的通路...")
                cache[cid_str] = get_pathways_by_compound(cid_str)
                time.sleep(0.2)  # 避免请求过快
            df_out.at[idx, 'kegg_map'] = cache[cid_str]
        else:
            # 无C号，保留 "-"
            pass
    
    # 保存
    if output_path is None:
        output_path = '2.xlsx'  # 默认输出为2.xlsx
    df_out.to_excel(output_path, index=False)
    print(f"✅ 生成完成，结果保存至: {output_path}")
    return df_out

if __name__ == "__main__":
    # 使用示例
    df = generate_kegg_annotation_from_c_number('1.xlsx', '2.xlsx')
    print(df.head())