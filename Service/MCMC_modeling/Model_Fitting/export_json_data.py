import json
import sys
import os

import csv
import TinyLensGpu.Profile.util as util
import numpy as np
from ..ModelComponentClass import ModelComponent

def csv_to_json(csv_filepath, json_filepath):
    # 检查CSV文件是否存在
    if not os.path.exists(csv_filepath):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected CSV file path: {csv_filepath}")
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    # 读取CSV文件
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    
    # 转换数据为浮点数
    e1_lens = float(data[1]["median"])
    e2_lens = float(data[2]["median"])
    e1_source = float(data[7]["median"])
    e2_source = float(data[8]["median"])

    # 调用util.ellipticity2phi_q并转换为float
    phi_lens, ell_lens = util.ellipticity2phi_q(e1_lens, e2_lens)
    phi_source, ell_source = util.ellipticity2phi_q(e1_source, e2_source)
    phi_lens = float(phi_lens*180/np.pi)
    phi_source = float(phi_source*180/np.pi)
    ell_lens = float(ell_lens)
    phi_source = float(phi_source)
    ell_source = float(ell_source)
    
    # 转换数据为JSON格式
    result = {
        "pixscale": 0.05,
        "components": [
            {
                "plane": "source",
                "size": float(data[5]["median"]),
                "ell": 1/ell_source,
                "ang": phi_source,
                "x": float(data[9]["median"]),
                "y": float(data[10]["median"]),
                "n_sersic": float(data[6]["median"]),
            },
            {
                "plane": "lens",
                "theta_e": float(data[0]["median"]),
                "ell": 1/ell_lens,
                "ang": phi_lens,
                "x": 0.0,
                "y": 0.0
            }
        ]
    }

    # 写入JSON文件
    with open(json_filepath, mode='w') as json_file:
        json.dump(result, json_file, indent=4)
    print(f"JSON file saved to: {json_filepath}")

def export_data_params():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建基于脚本目录的路径
    csv_filepath = os.path.join(script_dir, "output", "result_summary.csv")
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_filepath):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected CSV file path: {csv_filepath}")
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    # 转换数据为浮点数
    e1_lens = float(data[1]["median"])
    e2_lens = float(data[2]["median"])
    e1_source = float(data[7]["median"])
    e2_source = float(data[8]["median"])

    # 调用util.ellipticity2phi_q并转换为float
    phi_lens, ell_lens = util.ellipticity2phi_q(e1_lens, e2_lens)
    phi_source, ell_source = util.ellipticity2phi_q(e1_source, e2_source)
    phi_lens = float(phi_lens*180/np.pi)
    phi_source = float(phi_source*180/np.pi)
    ell_lens = float(ell_lens)
    phi_source = float(phi_source)
    ell_source = float(ell_source)

    # 转换数据为ModelComponent格式
    lens_component = ModelComponent(
        plane="lens",
        theta_e=float(data[0]["median"]),
        ell=1/ell_lens,
        ang=phi_lens,
        x=0.0,
        y=0.0
    )
    source_component = ModelComponent(
        plane="source",
        size=float(data[5]["median"]),
        ell=1/ell_source,
        ang=phi_source,
        x=float(data[9]["median"]),
        y=float(data[10]["median"]),
        n_sersic=float(data[6]["median"]),
    )
    # 去掉None值
    components = [source_component,lens_component]
    components = [comp.model_dump(exclude_none=True) for comp in components]
    return components


if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # 添加项目根目录到Python路径，以便能够导入TinyLensGpu模块
    project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
    sys.path.append(project_root)
    
    # 处理直接运行时的导入问题，临时替换相对导入
    import sys
    from importlib import import_module
    
    # 动态导入ModelComponentClass
    try:
        ModelComponentClass = import_module('Service.MCMC_modeling.ModelComponentClass')
        ModelComponent = ModelComponentClass.ModelComponent
    except Exception as e:
        print(f"动态导入ModelComponentClass失败: {e}")
        # 如果动态导入失败，尝试直接导入
        from ..ModelComponentClass import ModelComponent
    
    # 构建基于脚本目录的路径
    csv_filepath = os.path.join(script_dir, "output", "result_summary.csv")
    json_filepath = os.path.join(script_dir, "output", "result.json")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    
    csv_to_json(csv_filepath, json_filepath)
    print("模型输出完毕。")
