import io
import sys
import os
import gzip
import base64
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter
from GGSL_UNet.extract_background import process_ordinary_fits_and_extract_background
import matplotlib.pyplot as plt
from PIL import Image

extract_background_app = APIRouter()

@extract_background_app.post("/extract_background")
async def extract_background_post(file: UploadFile = File()):
    print("收到文件:", file.filename)
    try:
        # 记录原始工作目录
        orig_cwd = os.getcwd()
        # 切换到 GGSL_UNet 目录
        ggsl_unet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../GGSL_UNet'))
        os.chdir(ggsl_unet_dir)
        try:
            background = process_ordinary_fits_and_extract_background(
                file.file,
                output_fits_path='../Service/output/extracted_background.fits',
                save_background=True
            )
        except FileNotFoundError as fnf_err:
            print("模型文件缺失:", str(fnf_err))
            return {"status": "error", "message": f"模型文件缺失: {str(fnf_err)}"}
        except Exception as e:
            print("处理文件时出错:", str(e))
            return {"status": "error", "message": str(e)}
        finally:
            # 切回原始工作目录
            os.chdir(orig_cwd)
       # 归一化为 float32
        arr = (background - np.min(background)) / (np.max(background) - np.min(background) + 1e-8)
        arr = arr.astype(np.float32)
        # 压缩并编码
        arr_bytes = arr.tobytes()
        compressed = gzip.compress(arr_bytes)
        b64_data = base64.b64encode(compressed).decode('utf-8')
        # fits_file = fits.open('../Service/output/extracted_background.fits')
        return {
            "status": "success",
            "image_data": b64_data,
            "width": arr.shape[1],
            "height": arr.shape[0],
            # "fits_file": fits_file
        }
    except Exception as e:
        print("处理文件时出错:", str(e))
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print("处理文件时出错:", str(e))
        return {"status": "error", "message": str(e)}

