import csv  
import pandas as pd
#from plot import calculate_kl_divergence  
import os
from metrics import calculate_kl_divergence

#将txt读入到csv
def get_generatecsv():
    # Initialize a list to store rows  
    
# Read the data from the text file  
    input_file = r"Res_fullSimulation\\outputGPT4_zero.txt"
     # 输出文件路径
    output_dir = r"Res_fullSimulation\\CSV\\zeroshot"
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    output_file = os.path.join(output_dir, 'youth_zeroGPT_4.csv') 
    # Read the text file line by line  
   # 读取输入文件
    rows = []
   # Read the text file line by line  
    with open(input_file, "r") as file:
        for line in file:
            line = line.strip()  # 去除多余空格和换行符
            # 检查行是否以 "[" 开头，但不一定以 "]" 结尾
            if line.startswith("["):
                # 去掉开头的 "["，并处理可能存在的尾部逗号或其他字符
                line = line[1:].rstrip(",] \n")
                # 分割字符串为列表，并去除每个元素的多余引号
                values = [value.strip().strip("'").strip('"') for value in line.split(",")]
                rows.append(values)

    # Write the rows into a CSV file  
    column_names = ["Action", "Documentary", "Thriller", "Comedy"]  # Replace with your desired column names  

    # Write the rows into a CSV file  
    with open(output_file, "w", newline="") as csvfile:  
        writer = csv.writer(csvfile)  
        # Write the column names first  
        writer.writerow(column_names)  
        # Write the data rows  
        writer.writerows(rows)  

    print(f"Data has been successfully written to {output_file}")  

def drop():
    input_file = r"data_prepro\\Data\\responses_index.csv"
    col = ["Action", "Documentary", "Thriller", "Comedy"]
    df = pd.read_csv(input_file,usecols=col)
    df.dropna(subset=col,inplace=True)
    df.to_csv(r"Res_fullSimulation\\CSV\\cleaned_groundTruth.csv")

def randomDeal(input_file, output_file):
    df = pd.read_csv(input_file)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_csv(output_file, index=False)
    print(f"数据已成功打乱并保存到 {output_file}")
#将真实值和生成数据修改列名进行拼接
def merge():
    input_file_groundTruth = r"Res_fullSimulation\\CSV\\cleaned_groundTruth.csv"
    random_groundTruth =  r"Res_fullSimulation\\CSV\\random_groundTruth.csv"
    randomDeal(input_file_groundTruth,random_groundTruth)
    # 定义输入和输出文件路径  
    input_file1 = r"Res_fullSimulation\\CSV\\zeroshot\\youth_zeroGPT_4.csv"   # 第一个CSV文件  
    input_file2 = r"Res_fullSimulation\\CSV\\random_groundTruth.csv" # 第二个CSV文件  
    output_file = r"Res_fullSimulation\\CSV\\mergedYouthzeroGPT_4.csv"  # 合并后的CSV文件  

    columns_from_file1 = ["Action", "Documentary", "Thriller", "Comedy"]  # 从第一个文件中抽取的列  
    columns_from_file2 = ["Action", "Documentary", "Thriller", "Comedy"]  # 从第二个文件中抽取的列  

    # 定义新的列名  
    new_columns_file1 = {"Action": "GAction", "Documentary": "GDocumentary", "Thriller": "GThriller", "Comedy": "GComedy"}  # 第一个文件的列重命名  
    new_columns_file2 = {"Action": "TAction", "Documentary": "TDocumentary", "Thriller": "TThriller", "Comedy": "TComedy"}
    # 读取第一个CSV文件并抽取指定列  
    df1 = pd.read_csv(input_file1, usecols=columns_from_file1)  
    # 重命名列  
    df1.rename(columns=new_columns_file1, inplace=True)  

    # 读取第二个CSV文件并抽取指定列  
    df2 = pd.read_csv(input_file2, usecols=columns_from_file2)  
    df2.rename(columns=new_columns_file2, inplace=True) 

   
    # 获取两个文件的最短长度  
    min_length = min(len(df1), len(df2))  

    # 截取两个数据框到最短长度  
    df1 = df1.head(min_length)  
    df2 = df2.head(min_length)  

    # 合并两个数据框  
    merged_df = pd.concat([df1, df2], axis=1)  

    # 将合并后的数据写入新的CSV文件  
    merged_df.to_csv(output_file, index=False)  

    print(f"数据已成功合并并保存到 {output_file}")  

def cal():
    input_file = r"/home/cyyuan/Data/Trell social media usage/merged33zero.csv"
    with open(input_file,'r') as infile:
        reader = csv.DictReader(infile)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0

        for i, row in enumerate(reader):
            if row['fenvir'] == row['renvir']:
                cnt1 = cnt1 + 1
            if row['fheal'] == row['rheal']:
                cnt2 = cnt2 + 1
            if row['fcity'] == row['rcity']:
                cnt3 = cnt3 + 1
        acc1 = cnt1/(i+1)
        acc2 = cnt2/(i+1)
        acc3 = cnt3/(i+1)

        print(acc1,acc2,acc3)
    return
import numpy as np


def kl_divergence(p, q):  
    # 避免 q 为 0 的情况  
    q = np.where(q == 0, 1e-10, q)  
    # 计算 KL 散度，同时避免无效值  
    return np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))  

def compute_all_kl_divergences():
    file_path = "Res_fullSimulation\\CSV\\mergedYouthzero3.1.csv"
    df = pd.read_csv(file_path)
    # 确保数据类型为数值类型
    df['GAction'] = pd.to_numeric(df['GAction'], errors='coerce')
    df['TAction'] = pd.to_numeric(df['TAction'], errors='coerce')
    df['GDocumentary'] = pd.to_numeric(df['GDocumentary'], errors='coerce')
    df['TDocumentary'] = pd.to_numeric(df['TDocumentary'], errors='coerce')
    df['GThriller'] = pd.to_numeric(df['GThriller'], errors='coerce')
    df['TThriller'] = pd.to_numeric(df['TThriller'], errors='coerce')
    df['GComedy'] = pd.to_numeric(df['GComedy'], errors='coerce')
    df['TComedy'] = pd.to_numeric(df['TComedy'], errors='coerce')
    df = df.dropna(subset=['GComedy', 'TComedy'])
    
    # 删除全为0的行
    df = df[(df['GComedy'] != 0) & (df['TComedy'] != 0)]
    columns = ["GAction", "TAction", "GDocumentary", "TDocumentary", "GThriller", "TThriller", "GComedy", "TComedy"]
    kl_results = {}
    for i in range(0, len(columns), 2):
        col1, col2 = columns[i], columns[i+1]
        kl_div = calculate_kl_divergence(df[col1], df[col2])
        kl_results[f"{col1} vs {col2}"] = kl_div
# 输出结果
    for key, value in kl_results.items():
        print(f"{key}: KL Divergence = {value}")
    

# 文件路径  
if __name__ == "__main__":
    # get_generatecsv()
    # drop()
    # merge()
    #calculate_kl_for_columns(r"/home/cyyuan/Data/Trell social media usage/merged33zero.csv")
    # drop()
    compute_all_kl_divergences()