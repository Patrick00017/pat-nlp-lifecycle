import json
import yaml
from typing import Annotated, Any, Dict, List, Optional, TypedDict

def load_config(config_path="config.yaml"):
    """Load database configuration from a YAML file."""
    config = {}
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def material_part_count(material):
    # material = "N.-.-.7.N"
    parts = material.split('.')
    count = sum(1 for part in parts if part != '-')
    return count

def is_material_equal(full_material, material, part):
    if part == "DF":
        return full_material == material
    elif part == "SF1":
        full_material_sub_part = full_material.split(".") # 5 parts
        material_sub_part = material.split("/")
        return full_material_sub_part[1] == material_sub_part[0] and full_material_sub_part[2] == material_sub_part[1]
    elif part == "SF2":
        full_material_sub_part = full_material.split(".") # 5 parts
        material_sub_part = material.split("/")
        return full_material_sub_part[3] == material_sub_part[0] and full_material_sub_part[4] == material_sub_part[1]
    elif part == "LS0":
        return full_material[0] == material

def convert_lifecycle_to_markdown(data):
    """
    将SetGlueSF*数据转换为Markdown格式
    """
    markdown = []
    
    for lifecycle in data:
        markdown.append("## 🔄 生命周期\n")
        markdown.append("| 阶段 | 信息 | 时间 |")
        markdown.append("|------|------|------|")
        
        if 'ls0' in lifecycle:
            markdown.append(f"| **LS0** | `{lifecycle['ls0'].get('msg', 'N/A')}` | {lifecycle['ls0'].get('time', 'N/A')} |")
        if 'ms1' in lifecycle:
            markdown.append(f"| **MS1** | `{lifecycle['ms1'].get('msg', 'N/A')}` | {lifecycle['ms1'].get('time', 'N/A')} |")
        if 'ls1' in lifecycle:
            markdown.append(f"| **LS1** | `{lifecycle['ls1'].get('msg', 'N/A')}` | {lifecycle['ls1'].get('time', 'N/A')} |")
        if 'ms2' in lifecycle:
            markdown.append(f"| **MS2** | `{lifecycle['ms2'].get('msg', 'N/A')}` | {lifecycle['ms2'].get('time', 'N/A')} |")
        if 'ls2' in lifecycle:
            markdown.append(f"| **LS2** | `{lifecycle['ls2'].get('msg', 'N/A')}` | {lifecycle['ls2'].get('time', 'N/A')} |")
        if 'df' in lifecycle:
            markdown.append(f"| **DF** | `{lifecycle['df'].get('msg', 'N/A')}` | {lifecycle['df'].get('time', 'N/A')} |")
        if 'set_func' in lifecycle:
            markdown.append(f"| **Set Function** | `{lifecycle['set_func'].get('name', 'N/A')}` | {lifecycle['set_func'].get('time', 'N/A')} |")
        markdown.append(" \n\n --- \n")
    return "\n".join(markdown)


# handle markdown
def extract_h3_headings_v2(markdown_file: str) -> Dict[str, str]:
    """
    逐行解析Markdown文件，提取###标题及其内容
    遇到任何标题（#、##、###等）都视为当前内容的结束
    """
    with open(markdown_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = {}
    current_title = None
    current_content = []

    for line in lines:
        line_stripped = line.lstrip()  # 去除开头的空格，但保留缩进

        # 检查是否是任何级别的标题（以#开头）
        if line_stripped.startswith("#"):
            # 如果是###标题，开始新的记录
            if line_stripped.startswith("### "):
                # 如果之前有正在处理的标题，保存它
                if current_title:
                    result[current_title] = "".join(current_content).strip()

                # 开始新的标题
                current_title = line_stripped[4:].strip()  # 去掉'### '前缀
                current_content = []
            else:
                # 如果是其他级别的标题（# 或 ##），且当前正在收集内容
                if current_title:
                    # 保存当前###标题的内容
                    result[current_title] = "".join(current_content).strip()
                    # 重置，不再收集内容（因为新标题不是###）
                    current_title = None
                    current_content = []
                # 如果当前没有在收集内容，什么也不做
        elif current_title:
            # 如果当前在某个###标题下，添加内容
            current_content.append(line)

    # 保存最后一个标题
    if current_title:
        result[current_title] = "".join(current_content).strip()

    return result