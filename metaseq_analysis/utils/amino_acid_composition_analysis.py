import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ================================
# 1. 氨基酸映射和分类（固定）
# ================================
aa3_to_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Hyp': 'P', 'Hse': 'S', 'Orn': 'K', 'Cit': 'R', 'Abu': 'A', 'Nle': 'L',
    'Pca': 'Q'
}
acidic = {'D', 'E'}
basic = {'R', 'K', 'H'}
neutral = set(aa3_to_1.values()) - acidic - basic

full_to_3 = {
    'alanine': 'Ala', 'arginine': 'Arg', 'asparagine': 'Asn', 'aspartic acid': 'Asp',
    'cysteine': 'Cys', 'glutamine': 'Gln', 'glutamic acid': 'Glu', 'glycine': 'Gly',
    'histidine': 'His', 'isoleucine': 'Ile', 'leucine': 'Leu', 'lysine': 'Lys',
    'methionine': 'Met', 'phenylalanine': 'Phe', 'proline': 'Pro', 'serine': 'Ser',
    'threonine': 'Thr', 'tryptophan': 'Trp', 'tyrosine': 'Tyr', 'valine': 'Val',
    'hydroxyproline': 'Hyp', 'homoserine': 'Hse', 'ornithine': 'Orn', 'citrulline': 'Cit',
    'aminobutyric acid': 'Abu', 'norleucine': 'Nle', 'pyroglutamic acid': 'Pca'
}

special_compounds = {
    'glutathione': ['Glu', 'Cys', 'Gly'],
    's-nitrosoglutathione': ['Glu', 'Cys', 'Gly'],
    's-methylglutathione': ['Glu', 'Cys', 'Gly'],
    's-octylglutathione': ['Glu', 'Cys', 'Gly'],
    's-glutathionyl-l-cysteine': ['Glu', 'Cys', 'Gly', 'Cys'],
    'oxiglutatione': ['Glu', 'Cys', 'Gly'],
    'glutathionate': ['Glu', 'Cys', 'Gly'],
    'n-acetylcysteine amide': ['Cys'],
}


def normalize_name(name):
    s = name.lower().strip()
    s = re.sub(r'\([^)]*\)', '', s)  # 删除括号内容
    s = re.sub(r'hemisulfate salt.*$', '', s)
    s = re.sub(r'hydrochloride.*$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def extract_amino_acids(compound_name):
    name = normalize_name(compound_name)

    # 特殊化合物
    for key, aa_list in special_compounds.items():
        if key in name:
            return [aa3_to_1.get(aa, aa) for aa in aa_list]

    # 去除常见修饰前缀
    cleaned = re.sub(r'^(n-?acetyl-?|[ld]-|dl-|tert-butoxycarbonyl-|boc-|ac-|z-|cbz-)', '', name)
    cleaned = re.sub(r'-(oh|nh2|ome|oet|cooh|cho|ome|nhme|nme2)$', '', cleaned)
    cleaned = re.sub(r'^-', '', cleaned)

    # 1) 三字母代码
    triplets = re.findall(r'\b(?:' + '|'.join(aa3_to_1.keys()) + r')\b', cleaned, re.IGNORECASE)
    if triplets:
        return [aa3_to_1[t.capitalize()] for t in triplets]

    # 2) 单字母代码（仅当看起来是单字母序列时）
    if re.search(r'\b[ACDEFGHIKLMNPQRSTVWY]\b', cleaned, re.IGNORECASE):
        single_letters = re.findall(r'\b([ACDEFGHIKLMNPQRSTVWY])\b', cleaned, re.IGNORECASE)
        if single_letters:
            return [x.upper() for x in single_letters if x.upper() in aa3_to_1.values()]

    # 3) 全称匹配
    for full, code3 in full_to_3.items():
        if full in cleaned:
            return [aa3_to_1[code3]]

    # 4) 拆分连字符/空格
    parts = re.split(r'[-_\s]+', cleaned)
    found = []
    for p in parts:
        if p.capitalize() in aa3_to_1:
            found.append(aa3_to_1[p.capitalize()])
        elif p in full_to_3:
            found.append(aa3_to_1[full_to_3[p]])
        else:
            for full, code3 in full_to_3.items():
                if full in p:
                    found.append(aa3_to_1[code3])
                    break
    return found


def read_compound_file(file_path):
    """从文件读取化合物列表（每行一个）"""
    compounds = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                compounds.append(line)
    return compounds


def analyze_compound_list(compounds, output_prefix='analysis'):
    """
    分析化合物列表中的氨基酸组成并绘图
    :param compounds: list of str
    :param output_prefix: str
    """
    counter = Counter()
    unrecognized = []
    for comp in compounds:
        aas = extract_amino_acids(comp)
        if aas:
            counter.update(aas)
        else:
            unrecognized.append(comp)

    if unrecognized:
        print(f"\n警告：以下 {len(unrecognized)} 个化合物未识别出氨基酸：")
        for u in unrecognized[:20]:
            print(f"  {u}")
        if len(unrecognized) > 20:
            print(f"  ... 还有 {len(unrecognized) - 20} 个")

    if not counter:
        print("未识别到任何氨基酸，请检查规则或输入数据。")
        return

    all_aa = sorted(set(aa3_to_1.values()))
    counts = [counter.get(aa, 0) for aa in all_aa]
    df = pd.DataFrame({'AminoAcid': all_aa, 'Count': counts})
    df['Category'] = df['AminoAcid'].map(
        lambda aa: 'Acidic' if aa in acidic else ('Basic' if aa in basic else 'Neutral')
    )
    # 确保 Count 列为整数类型
    df['Count'] = df['Count'].astype(int)

    print("\n各氨基酸出现次数统计：")
    print(df.sort_values('Count', ascending=False).to_string(index=False))

    # 绘图1：三分类子图
    categories = ['Acidic', 'Basic', 'Neutral']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, cat in enumerate(categories):
        ax = axes[idx]
        subset = df[df['Category'] == cat].sort_values('Count', ascending=False)
        if subset.empty:
            ax.set_title(f'{cat} (no data)')
            ax.set_axis_off()
            continue
        ax.bar(subset['AminoAcid'], subset['Count'], color='steelblue')
        ax.set_title(f'{cat} Amino Acids')
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Occurrence Count')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.text(i, row['Count'] + 0.5, str(row['Count']), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_by_category.png', dpi=300)
    plt.show()

    # 绘图2：堆叠柱状图（使用字典映射避免 reindex 类型错误）
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    color_map = {'Acidic': '#ff9999', 'Basic': '#99cc99', 'Neutral': '#99ccff'}
    # 确定所有氨基酸的顺序（按类别和字母）
    order = sorted(all_aa, key=lambda aa: (df[df['AminoAcid'] == aa]['Category'].iloc[0], aa))
    bottom = np.zeros(len(order))
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        # 创建氨基酸->计数的字典
        cat_counts = dict(zip(cat_df['AminoAcid'], cat_df['Count']))
        # 按 order 顺序取值，缺失则为0
        cat_vals = [cat_counts.get(aa, 0) for aa in order]
        ax2.bar(order, cat_vals, label=cat, bottom=bottom, color=color_map[cat])
        bottom += np.array(cat_vals)
    ax2.set_xlabel('Amino Acid (1-letter)')
    ax2.set_ylabel('Occurrence Count')
    ax2.set_title('Total Amino Acid Occurrences')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_total_stacked.png', dpi=300)
    plt.show()
    print(f"\n图片已保存为 {output_prefix}_by_category.png 和 {output_prefix}_total_stacked.png")


def analyze_compound_file(file_path, output_prefix='analysis'):
    """
    从文件读取化合物列表并分析
    :param file_path: str, 化合物列表文件路径（每行一个化合物）
    :param output_prefix: str
    """
    compounds = read_compound_file(file_path)
    print(f"从 {file_path} 读取了 {len(compounds)} 个化合物")
    analyze_compound_list(compounds, output_prefix)


# ================================
# 2. 使用示例
# ================================
if __name__ == '__main__':
    # 请根据实际路径修改
    # analyze_compound_file('class0_compounds.txt', 'class0')

    # 或者直接传入列表测试
    # test_list = ['Tyr-Asp', 'Glutathione', 'Phe-Glu']
    # analyze_compound_list(test_list, 'test')

    pass