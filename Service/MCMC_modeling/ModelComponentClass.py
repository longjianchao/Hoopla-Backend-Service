from pydantic import BaseModel
from typing import List, Dict, Any

# 定义模型组件数据结构
class ModelComponent(BaseModel):
    plane: str
    # source平面字段
    size: float = None  # 可选字段
    n_sersic: float = None  # 可选字段
    # lens平面字段
    theta_e: float = None  # 可选字段
    # 共用字段
    ell: float
    ang: float
    x: float
    y: float

# 定义请求数据结构
class MCMCRequest(BaseModel):
    components: List[Dict[str, Any]]  # 使用Dict保持灵活性
    pixscale: float
    image_data: Any  # 图像数据可能是复杂结构，使用Any类型
