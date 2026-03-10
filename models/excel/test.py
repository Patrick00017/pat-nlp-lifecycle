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


# 数据预处理
def preprocess_data(df):
    """数据预处理：填充空值，处理合并单元格"""
    # 创建新DataFrame避免修改原数据
    df_processed = df.copy()

    # 向前填充序号、姓名、身份证号等关键信息（处理合并单元格）
    columns_to_fill = [
        "序号",
        "姓名",
        "身份证号码",
        "实际工作单位名称",
        "实际工作单位税号",
        "用工性质",
        "工资总额",
        "2025年实际工作天数",
        "代交五险一金合计",
    ]

    for col in columns_to_fill:
        if col in df_processed.columns:
            # 将空值向前填充（处理合并单元格的情况）
            df_processed[col] = df_processed[col].fillna(method="ffill")

    # 删除没有工资或工作单位的行
    df_processed = df_processed.dropna(subset=["工资", "实际工作单位名称"], how="all")

    # 确保工资列为数值类型
    if "工资" in df_processed.columns:
        df_processed["工资"] = pd.to_numeric(df_processed["工资"], errors="coerce")
        # 删除工资为0或空的行
        df_processed = df_processed.dropna(subset=["工资"])
        df_processed = df_processed[df_processed["工资"] > 0]

    print(f"预处理后，共 {len(df_processed)} 行有效数据")
    return df_processed


def merge_person_company_summary(df):
    # new_df = pd.DataFrame(
    #     columns=[
    #         "序号",
    #         "姓名",
    #         "身份证号码",
    #         "实际工作单位名称",
    #         "实际工作单位税号",
    #         "用工性质",
    #         "工资总额",
    #         "2025年实际工作天数",
    #         "代交五险一金合计",
    #         "工资小计",
    #         "工作月数",
    #         "平均月薪",
    #         "工作月份列表",
    #     ]
    # )

    cache = {}
    for index, row in df.iterrows():
        name = row["姓名"]
        company = row["实际工作单位名称"]
        key = name + company
        if key not in cache:
            cache[key] = row
            continue
        # if exist
        cache[key]["工资小计"] += row["工资小计"]

    arr = []
    for value in cache.values():
        arr.append(value)

    new_df = pd.DataFrame(
        arr,
        columns=[
            "序号",
            "姓名",
            "身份证号码",
            "实际工作单位名称",
            "实际工作单位税号",
            "用工性质",
            "工资总额",
            "2025年实际工作天数",
            "代交五险一金合计",
            "工资小计",
            "工作月数",
            "平均月薪",
            "工作月份列表",
        ],
    )
    print(new_df)
    return new_df


# 按人员-公司汇总工资（人员在该公司的工资小结）
def summarize_by_person_company(df):
    """按人员和工作单位汇总工资，计算每个人员在该公司的工资总额"""

    # 创建人员-公司唯一标识
    df["人员公司"] = df["姓名"] + "_" + df["实际工作单位名称"]

    # 按人员和工作单位分组汇总
    person_company_summary = (
        df.groupby(
            [
                "序号",
                "姓名",
                "身份证号码",
                "实际工作单位名称",
                "实际工作单位税号",
                "用工性质",
                "工资总额",
                "2025年实际工作天数",
                "代交五险一金合计",
            ]
        )
        .agg(
            {
                "工资": ["sum", "count", "mean"],  # 总额、次数、平均值
                "工作时间": lambda x: list(x),  # 列出工作月份
            }
        )
        .reset_index()
    )

    # 重命名列
    person_company_summary.columns = [
        "序号",
        "姓名",
        "身份证号码",
        "实际工作单位名称",
        "实际工作单位税号",
        "用工性质",
        "工资总额",
        "2025年实际工作天数",
        "代交五险一金合计",
        "工资小计",
        "工作月数",
        "平均月薪",
        "工作月份列表",
    ]

    # 按姓名和工资总额排序
    person_company_summary = person_company_summary.sort_values(
        ["姓名", "工资总额"], ascending=[True, False]
    ).reset_index(drop=True)

    # 统计
    person_company_summary = merge_person_company_summary(person_company_summary)
    # person_company_summary = person_company_summary.sort_values("序号").reset_index(
    #     drop=True
    # )

    return person_company_summary


# 按人员汇总（每个人员的总工资）
def summarize_by_person(df):
    """按人员汇总总工资"""

    person_summary = (
        df.groupby(["姓名", "身份证号码"])
        .agg(
            {
                "工资": "sum",
                "实际工作单位名称": lambda x: list(set(x)),  # 列出所有工作过的单位
            }
        )
        .rename(columns={"工资": "年度总工资", "实际工作单位名称": "工作单位列表"})
        .reset_index()
    )

    # 添加工作单位数量
    person_summary["工作单位数量"] = person_summary["工作单位列表"].apply(len)

    # 按年度总工资降序排序
    # person_summary = person_summary.sort_values(
    #     "年度总工资", ascending=False
    # ).reset_index(drop=True)

    return person_summary


# 按公司汇总（每个公司的总工资）
def summarize_by_company(df):
    """按公司汇总总工资"""

    company_summary = (
        df.groupby(["实际工作单位名称", "实际工作单位税号"])
        .agg(
            {
                "工资": "sum",
                "姓名": lambda x: list(set(x)),  # 列出在该公司工作过的所有人员
            }
        )
        .rename(columns={"工资": "公司总工资", "姓名": "工作人员列表"})
        .reset_index()
    )

    # 添加工作人员数量
    company_summary["工作人员数量"] = company_summary["工作人员列表"].apply(len)

    # 按公司总工资降序排序
    # company_summary = company_summary.sort_values(
    #     "公司总工资", ascending=False
    # ).reset_index(drop=True)

    return company_summary


# 生成详细的人员-公司工资报表
def generate_detailed_report(df):
    """生成详细的报表，包含每个人员在不同公司的工资明细"""

    # 创建人员-公司唯一标识
    df["人员公司"] = df["姓名"] + "_" + df["实际工作单位名称"]

    # 先按人员-公司分组，获取汇总信息
    person_company = (
        df.groupby(
            [
                "序号",
                "姓名",
                "身份证号码",
                "实际工作单位名称",
                "实际工作单位税号",
                "用工性质",
            ]
        )
        # .agg({"工资": ["sum", "count"], "工作时间": lambda x: sorted(list(x))})
        .agg({"工资": ["sum", "count"], "工作时间": lambda x: list(x)}).reset_index()
    )

    person_company.columns = [
        "序号",
        "姓名",
        "身份证号码",
        "工作单位名称",
        "单位税号",
        "用工性质",
        "工资总额",
        "工作月数",
        "工作月份",
    ]

    # 添加月份范围信息
    person_company["工作月份范围"] = person_company["工作月份"].apply(
        lambda x: f"{x[0]} 至 {x[-1]}" if len(x) > 1 else x[0]
    )

    # 按姓名排序
    # person_company = person_company.sort_values(
    #     ["姓名", "工作月数"], ascending=[True, False]
    # ).reset_index(drop=True)

    return person_company


# 输出结果
def save_summaries(
    person_company_summary,
    person_summary,
    company_summary,
    detailed_report,
    output_file,
):
    """将汇总结果保存到Excel文件"""

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # 人员-公司工资小结（主要报表）
        person_company_summary.to_excel(
            writer, sheet_name="人员-公司工资小结", index=False
        )

        # 详细报表（包含工作月份）
        detailed_report.to_excel(writer, sheet_name="详细报表-含月份", index=False)

        # 人员总工资汇总
        person_summary.to_excel(writer, sheet_name="人员年度总工资", index=False)

        # 公司总工资汇总
        company_summary.to_excel(writer, sheet_name="公司总工资", index=False)

        # 添加统计信息
        stats_data = {
            "统计项": [
                "总人数",
                "总单位数",
                "总工资额",
                "平均每人工资",
                "平均每单位工资",
            ],
            "数值": [
                len(person_summary),
                len(company_summary),
                person_summary["年度总工资"].sum(),
                person_summary["年度总工资"].mean(),
                company_summary["公司总工资"].mean(),
            ],
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name="统计信息", index=False)

    print(f"汇总结果已保存到: {output_file}")


# 显示示例结果
def display_samples(person_company_summary):
    """显示几个示例结果"""
    print("\n=== 人员-公司工资小结示例（前10条）===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)

    # 选择几列显示
    display_cols = ["姓名", "工作单位名称", "工资总额", "工作月数", "平均月薪"]
    print(person_company_summary[display_cols].head(10))

    # 显示单个人员的完整记录示例
    if len(person_company_summary) > 0:
        sample_person = person_company_summary.iloc[0]["姓名"]
        print(f"\n=== 人员 '{sample_person}' 的工资小结 ===")
        person_records = person_company_summary[
            person_company_summary["姓名"] == sample_person
        ]
        print(
            person_records[["姓名", "工作单位名称", "工资总额", "工作月数", "平均月薪"]]
        )


# 主函数
def main():
    # 文件路径
    input_file = "绿州名单1-12总raw.xlsx"
    output_file = "step1.xlsx"

    # 检查文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 找不到文件 {input_file}")
        return

    # 读取数据
    print("正在读取Excel文件...")
    df = read_excel_file(input_file)
    if df is None:
        return

    # 数据预处理
    print("正在预处理数据...")
    df_processed = preprocess_data(df)

    # print(df_processed)

    # 按人员-公司汇总（主要报表）
    print("正在计算每个人员在各公司的工资小结...")
    person_company_summary = summarize_by_person_company(df_processed)

    # 生成详细报表（包含工作月份）
    print("正在生成详细报表...")
    detailed_report = generate_detailed_report(df_processed)

    # 按人员汇总总工资
    print("正在计算人员年度总工资...")
    person_summary = summarize_by_person(df_processed)

    # 按公司汇总总工资
    print("正在计算公司总工资...")
    company_summary = summarize_by_company(df_processed)

    # 显示示例
    # display_samples(person_company_summary)

    # 输出总体统计
    print(f"\n=== 总体统计 ===")
    print(f"总记录数: {len(person_company_summary)} 条人员-公司记录")
    print(f"涉及人员: {len(person_summary)} 人")
    print(f"涉及单位: {len(company_summary)} 家")
    print(f"年度总工资额: {person_summary['年度总工资'].sum():,.2f} 元")

    # 保存结果
    print("\n正在保存结果...")
    save_summaries(
        person_company_summary,
        person_summary,
        company_summary,
        detailed_report,
        output_file,
    )

    print("处理完成!")


if __name__ == "__main__":
    main()
