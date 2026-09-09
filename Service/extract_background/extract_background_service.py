import io
import sys
import os
import gzip
import base64
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import uvicorn
from fastapi import FastAPI, File, UploadFile, APIRouter
from .extract_background import process_ordinary_fits_and_extract_background
import matplotlib.pyplot as plt
from PIL import Image

extract_background_app = APIRouter()

@extract_background_app.post("/extract_background")
async def extract_background_post(file: UploadFile = File()):
    print("Received file:", file.filename)
    try:
        # Set up the temporary and output directories
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        output_dir = os.path.join(os.path.dirname(__file__), "../output")
        output_path = os.path.join(output_dir, "extracted_background.fits")
        
        # Process the file and extract the background image and prediction result data
        background, pred_data = process_ordinary_fits_and_extract_background(
            file.file,
            output_fits_path=output_path,
            save_background=True,
            temp_dir=temp_dir
        )
        
        arr = background.astype(np.float64)
        # Compress and encode
        arr_bytes = arr.tobytes()
        compressed = gzip.compress(arr_bytes)
        b64_data = base64.b64encode(compressed).decode('utf-8')
        
        # Import the calculate_real_chi function
        from GGSL_UNet.plot_image import calculate_real_chi
        
        # Use calculate_real_chi from GGSL_UNet to compute the chi-square values
        chi_square_1, chi_square_2 = calculate_real_chi(pred_data)
        
        
        return {
            "status": "success",
            "image_data": b64_data,
            "width": arr.shape[1],
            "height": arr.shape[0],
            "chi_square_1": float(chi_square_1[0]),
            "chi_square_2": float(chi_square_2[0])
        }
    except FileNotFoundError as fnf_err:
        print("Model file missing:", str(fnf_err))
        return {"status": "error", "message": f"Model file missing: {str(fnf_err)}"}
    except Exception as e:
        print("Error processing file:", str(e))
        return {"status": "error", "message": str(e)}

