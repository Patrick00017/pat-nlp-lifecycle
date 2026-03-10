import pandas as pd
import numpy as np
from pathlib import Path


# 读取Excel文件
def read_excel_file(file_path):
    """读取Excel文件并返回DataFrame"""
    try:
        df = pd.read_excel(file_path, sheet_name=0, header=0)
        print(f"成功读取文件，共 {len(df)} 行数据")
        return df
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


# 主函数
def main():
    # 文件路径
    input_file1 = "绿州名单1-12总raw.xlsx"
    input_file2 = "step2_merge.xlsx"

    # 读取数据
    print("正在读取Excel文件...")
    raw_df = read_excel_file(input_file1)
    if raw_df is None:
        return

    # 读取数据
    print("正在读取Excel文件...")
    new_df = read_excel_file(input_file2)
    if new_df is None:
        return

    raw_cache = {}
    for index, row in raw_df.iterrows():
        id = str(row["身份证号码"]).strip()
        if id not in raw_cache:
            raw_cache[id] = row["工资总额"]

    new_cache = {}
    for index, row in new_df.iterrows():
        id = str(row["身份证号码"]).strip()
        if id not in new_cache:
            new_cache[id] = row["工资总额"]

    for key in raw_cache.keys():
        if key not in new_cache:
            print(f"没有身份证id: {key}")
        print(f"{key} -> {raw_cache[key]} : {new_cache[key]}")


if __name__ == "__main__":
    main()
