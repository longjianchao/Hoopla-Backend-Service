import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from fastapi import APIRouter, Request
import threading
import io
import base64

from MCMC_modeling.Model_Fitting.load_json_data import update_model_config
from MCMC_modeling.Model_Fitting.run_model_from_yaml import run_model
from MCMC_modeling.Model_Fitting.export_json_data import export_data_params, csv_to_json
import json
import os
from .ModelComponentClass import MCMCRequest

# 共享日志缓冲区（被 WebSocket 读取）
model_log_buffer = io.StringIO()
def run_model_thread():
    """后台线程执行建模任务，并捕获 print 输出"""
    global model_log_buffer
    old_stdout = sys.stdout
    sys.stdout = model_log_buffer
    try:
        run_model()  # 执行实际建模
        print("✅ 模型运行结束。")
        # 模型运行完成后执行导出操作
        try:
            print("正在导出结果数据...")
            csv_to_json('output/result_summary.csv', 'output/result.json')
            print("✅ 结果导出完成。")
        except Exception as export_error:
            print(f"❌ 结果导出出错: {export_error}")
    except Exception as e:
        print(f"❌ 模型运行出错: {e}")
    finally:
        sys.stdout = old_stdout


mcmc_app = APIRouter()
@mcmc_app.post("/mcmc_modeling")
async def mcmc_modeling_post(request: Request):
    try:
        # 获取请求体数据
        model_data = await request.json()
        # 打印接收到的参数概览
        print("接收到的MCMC请求数据:")
        # 详细解析components数组
        components = model_data.get('components', [])
        pixscale = model_data.get('pixscale', 0)        
        print(f"总组件数量: {len(components)}")
        print(f"Pixel Scale: {pixscale}")        
        # 区分并详细打印source和lens组件
        source_components = []
        lens_components = []
        
        for i, component in enumerate(components):
            plane_type = component.get('plane', 'unknown')
            print(f"组件 #{i+1} (类型: {plane_type}):")
            
            # 打印共用字段
            print(f"  椭圆率 (ell): {component.get('ell', 'N/A')}")
            print(f"  角度 (ang): {component.get('ang', 'N/A')}")
            print(f"  X坐标 (x): {component.get('x', 'N/A')}")
            print(f"  Y坐标 (y): {component.get('y', 'N/A')}")
            
            # 根据平面类型打印特定字段
            if plane_type == 'source':
                print(f"  大小 (size): {component.get('size', 'N/A')}")
                print(f"  Sersic指数 (n_sersic): {component.get('n_sersic', 'N/A')}")
                source_components.append(component)
            elif plane_type == 'lens':
                print(f"  Einstein半径 (theta_e): {component.get('theta_e', 'N/A')}")
                lens_components.append(component)
            
        print(f"Source组件数量: {len(source_components)}")
        print(f"Lens组件数量: {len(lens_components)}")

        # 将lens_components和source_components转换成字典
        lens_params = lens_components[0] if len(lens_components) > 0 else {}
        source_params = source_components[0] if len(source_components) > 0 else {}
        
        if lens_params and source_params:
            update_model_config(lens_params=lens_params, source_params=source_params, pixscale=pixscale)
        else:
            print("警告: 没有找到有效的lens或source组件，无法更新配置文件")

        # 清空旧日志
        model_log_buffer.seek(0)
        model_log_buffer.truncate(0)

        # 启动线程运行模型（非阻塞）
        thread = threading.Thread(target=run_model_thread)
        thread.daemon = True
        thread.start()
        
        # 线程已在后台运行，立即返回响应给前端
        # 模型运行结果将通过WebSocket实时推送
        
        # run_model()
        # components = export_data_params()
        # # print(components)
        # csv_to_json('output/result_summary.csv', 'output/result.json')

        # 返回响应，表示任务已提交
        return {
            "status": "success",
            "message": "建模任务已提交到后台运行，请等待完成通知。"
        }
        
    except Exception as e:
        print(f"处理MCMC请求时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误栈
        return {"status": "error", "message": f"处理MCMC请求时出错: {str(e)}"}


@mcmc_app.get("/modeling_result")
async def get_modeling_result():
    """获取建模结果数据"""
    try:
        # 定义文件路径
        csv_file = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "result_summary.csv")
        json_file = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "result.json")
        result_img_path = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "model_results.png")
        corner_img_path = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "model_corner.png")
        
        # 检查CSV结果文件是否存在
        if not os.path.exists(csv_file):
            print(f"结果CSV文件不存在: {csv_file}")
            return {"status": "error", "message": "建模结果CSV文件不存在，请确认建模已完成"}
        
        try:
            # 使用csv_to_json函数生成/更新JSON文件
            csv_to_json(csv_file, json_file)
            
            # 调用export_data_params函数获取components
            components = export_data_params()
            
            # 读取并转换图片文件为base64格式
            model_result = None
            model_corner = None
            
            if os.path.exists(result_img_path):
                with open(result_img_path, "rb") as img_file:
                    model_result = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"已读取model result: {result_img_path}")
            else:
                print(f"model result不存在: {result_img_path}")
                
            if os.path.exists(corner_img_path):
                with open(corner_img_path, "rb") as img_file:
                    model_corner = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"已读取corner plot: {corner_img_path}")
            else:
                print(f"corner plot不存在: {corner_img_path}")
            
            return {
                "status": "success",
                "components": components,
                "model_result": model_result,
                "model_corner": model_corner
            }
        except Exception as export_error:
            print(f"处理结果数据时出错: {export_error}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"处理结果数据时出错: {str(export_error)}"
            }
    except Exception as e:
        print(f"获取建模结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"获取建模结果时出错: {str(e)}"}
