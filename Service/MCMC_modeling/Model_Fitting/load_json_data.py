import json
import tkinter as tk
from tkinter import filedialog
import TinyLensGpu.Profile.util as util
import yaml  # 用于处理 YAML 文件
from math import cos, sin, radians
import os
import numpy as np

def load_json_file(title="选择一个 JSON 文件", filetypes=[("JSON 文件", "*.json")]):
    """通用的加载 JSON 文件函数"""
    # root = tk.Tk()
    # root.withdraw()  # 隐藏主窗口
    # file_path = filedialog.askopenfilename(
    #     title=title,
    #     filetypes=filetypes
    # )
    file_path = "output/result.json"
    if not file_path:
        print("未选择文件。")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"JSON 文件解析错误: {file_path}")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
    return None

def convert_to_array(frontData):
    # 添加对frontData结构的检查
    if not isinstance(frontData, dict) or 'components' not in frontData:
        print("输入数据结构不正确，缺少components字段")
        return
    
    if not isinstance(frontData['components'], list):
        print("components应该是一个列表")
        return
    
    lens_params = next((c for c in frontData['components'] if isinstance(c, dict) and c.get('plane') == 'lens'), None)
    source_params = next((c for c in frontData['components'] if isinstance(c, dict) and c.get('plane') == 'source'), None)
    
    if not isinstance(lens_params, dict) or not isinstance(source_params, dict):
        print("无法找到有效格式的lens或source参数")
        print(f"lens_params类型: {type(lens_params)}, source_params类型: {type(source_params)}")
        return
    
    pixscale = frontData.get("pixscale", None)  # 从 JSON 数据中获取 pixscale
    pixscale = float(pixscale) if pixscale is not None else None  # 确保 pixscale 是浮点数
    if lens_params and source_params:
        print("Lens 参数:", lens_params)
        print("Source 参数:", source_params)
        print("Pixel Scale:", pixscale)
        update_model_config(lens_params, source_params, pixscale)
    else:
        print("未找到 lens 或 source 参数。")

def sanitize_data(data):
    """将数据转换为基本的 Python 数据类型"""
    return json.loads(json.dumps(data))

def safe_round(value, default=0.0, decimals=4):
    """安全地将值转换为浮点数并四舍五入"""
    try:
        # 先尝试将值转换为浮点数
        num_value = float(value)
        # 然后再进行四舍五入
        return round(num_value, decimals)
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        print(f"警告: 无法将 '{value}' 转换为数字，使用默认值 {default}")
        return round(default, decimals)

def update_model_config(lens_params, source_params, pixscale, config_file="model_config.yaml"):
    """更新 model_config.yaml 文件中的 lens_mass_list、source_light_list 和 pixscale"""
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，请确保文件已创建或提供正确路径。")
            return

        # 读取现有的 YAML 配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 添加参数类型检查
        if not isinstance(lens_params, dict):
            print(f"lens_params 应该是字典类型，但实际是 {type(lens_params)} 类型")
            return
        
        if not isinstance(source_params, dict):
            print(f"source_params 应该是字典类型，但实际是 {type(source_params)} 类型")
            return

        # 更新 lens_mass_list 中的参数
        lens_mass_list = config["model_components"]["lens_mass_list"]
        if lens_mass_list and len(lens_mass_list) > 0:
            e1, e2 = util.phi_q2_ellipticity(
                safe_round(lens_params.get("ang", 0.0)*np.pi/180, 0.0), 
                safe_round(lens_params.get("ell", 0.0), 0.0)
            )
            # 在这里因为前端的json文件中的ell都是大于1的，要转换成椭圆参数，所以得加上负号
            e1, e2 = -float(e1), -float(e2)  
            lens_mass_list[0]["params"]["theta_E"] = {
                "prior_type": "uniform",
                "prior_settings": [safe_round(lens_params.get("theta_e", 0.0), 0.0) - 0.2,
                                   safe_round(lens_params.get("theta_e", 0.0), 0.0) + 0.2],
                "limits": [0.0, 10.0],
                "fixed": False
            }
            lens_mass_list[0]["params"]["e1"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(e1, 0.0), 0.1],
                "limits": [-1.0, 1.0],
                "fixed": False
            }
            lens_mass_list[0]["params"]["e2"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(e2, 0.0), 0.1],
                "limits": [-1.0, 1.0],
                "fixed": False
            }
            lens_mass_list[0]["params"]["center_x"] = {
                # "prior_type": "gaussian",
                # "prior_settings": [safe_round(lens_params.get("x", 0.0), 0.0), 0.1],
                "fixed": True,
                "fixed_value": 0.0
                # "limits": [-3.0, 3.0],
            }
            lens_mass_list[0]["params"]["center_y"] = {
                # "prior_type": "gaussian",
                # "prior_settings": [safe_round(lens_params.get("y", 0.0), 0.0), 0.1],
                "fixed": True,
                "fixed_value": 0.0
                # "limits": [-3.0, 3.0],
            }

        # 更新 source_light_list 中的参数
        source_light_list = config["model_components"]["source_light_list"]
        if source_light_list and len(source_light_list) > 0:
            e1, e2 = util.phi_q2_ellipticity(
                safe_round(source_params.get("ang", 0.0)*np.pi/180, 0.0), 
                safe_round(source_params.get("ell", 0.0), 0.0)
            )
            # 在这里因为前端的json文件中的ell都是大于1的，要转换成椭圆参数，所以得加上负号
            e1, e2 = -float(e1), -float(e2) 
            R_sersic = safe_round(source_params.get("size", 0.0), 0.0)
            source_light_list[0]["params"]["R_sersic"] = {
                "prior_type": "uniform",
                "prior_settings": [safe_round(max(R_sersic - 0.3, 0.01), 0.0),
                                   safe_round(R_sersic, 0.0) + 0.3],
                "limits": [0.0, 5.0],
                "fixed": False
            }
            source_light_list[0]["params"]["e1"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(e1, 0.0), 0.1],
                "limits": [-1.0, 1.0],
                "fixed": False
            }
            source_light_list[0]["params"]["e2"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(e2, 0.0), 0.1],
                "limits": [-1.0, 1.0],
                "fixed": False
            }
            source_light_list[0]["params"]["center_x"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(source_params.get("x", 0.0), 0.0), 0.1],
                "limits": [-3.0, 3.0],
                "fixed": False
            }
            source_light_list[0]["params"]["center_y"] = {
                "prior_type": "gaussian",
                "prior_settings": [safe_round(source_params.get("y", 0.0), 0.0), 0.1],
                "limits": [-3.0, 3.0],
                "fixed": False
            }
            n_sersic = safe_round(source_params.get("n_sersic", 0.0), 0.0)
            source_light_list[0]["params"]["n_sersic"] = {
                "prior_type": "uniform",
                "prior_settings": [safe_round(max(n_sersic - 0.5, 0.3), 0.0),
                                   safe_round(n_sersic, 0.0) + 0.5],
                "limits": [0.3, 6.0],
                "fixed": False
            }

        # 更新 dataset 中的 pixscale
        if pixscale is not None:
            config["dataset"]["pixel_scale"] = safe_round(pixscale, 0.0)

        # 将数据转换为基本类型
        sanitized_config = sanitize_data(config)

        # 将更新后的配置写回 YAML 文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(
                sanitized_config,
                f,
                allow_unicode=True,
                default_flow_style=False,  # 使用块样式
                sort_keys=False  # 保持字段顺序
            )
        print(f"配置文件 {config_file} 已成功更新。")
    except Exception as e:
        print(f"更新配置文件时发生错误: {e}")

# 仅当直接运行此脚本时才执行以下代码
if __name__ == "__main__":
    # 设置工作目录为脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    frontData = load_json_file(title="选择前端参数文件")
    if frontData:
        convert_to_array(frontData)

# e1, e2 = util.phi_q2_ellipticity(90/180*np.pi,0.9)
# # ang, ell = util.ellipticity2phi_q(0.31285601982951255,0.00038621935876378776)
# ang, ell = util.ellipticity2phi_q(e1,e2)
# print(f"e1,e2=",e1,e2)
# print(f"ang,ell=",ang*180/np.pi,ell)
