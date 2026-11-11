import pandas as pd

def get_data_by_geneid(id_list, df_data):
    '''注意df_data中的id作为索引'''
    bool_list = []
    for i in df_data.index.to_list():
        if i in id_list:
            bool_list.append(True)
        else:
            bool_list.append(False)

    results = df_data.loc[bool_list, :]
    return results


'''
测试代码
import os 

file_path = r'D:\MachineLearning\3_MyStudy\3.3-周梦迪_CsMAPKKK1\数据分析\mapkkk1_data_anlaysis\1-基因的锁定和mapkkk1.1调控黄瓜耐冷性的机制'
os.chdir(file_path)

csv_id_file = 'merged_different_genes.csv'
csv_data_file = 'data1_All_gene_counts.csv'

df = get_data_by_geneid(file_path, csv_id_file, csv_data_file)
df.to_csv('all_differential_genes.csv')
'''
