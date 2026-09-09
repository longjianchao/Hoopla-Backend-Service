import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from fastapi import APIRouter, Request, UploadFile, File, Form
import threading
import io
import base64
import shutil
import time

from MCMC_modeling.Model_Fitting.load_json_data import update_model_config
# from MCMC_modeling.Model_Fitting.load_data_cao_cfg import update_model_config
from MCMC_modeling.Model_Fitting.run_model_from_yaml import run_model
from MCMC_modeling.Model_Fitting.export_json_data import export_data_params, csv_to_json
import json
import os
from .ModelComponentClass import MCMCRequest

# Shared log buffer (read by the WebSocket)
model_log_buffer = io.StringIO()
def run_model_thread():
    """Run the modeling task in a background thread and capture print output"""
    global model_log_buffer
    old_stdout = sys.stdout
    sys.stdout = model_log_buffer
    try:
        run_model()  # Execute the actual modeling
        print("✅ Modeling finished.")
        # Perform the export after the model run completes
        try:
            print("Exporting result data...")
            csv_to_json('output/result_summary.csv', 'output/result.json')
            print("✅ Results exported.")
        except Exception as export_error:
            print(f"❌ Error exporting results: {export_error}")
        
        # Save the sampler_output.jsonl file
        try:
            print("Saving sampler_output.jsonl...")
            output_dir = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output")
            jsonl_path = os.path.join(output_dir, "sampler_output.jsonl")
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                log_content = model_log_buffer.getvalue()
                for line in log_content.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('Status') or line.startswith('Finished') or 
                                 'likelihood:' in line or line.startswith('Lens modeling with') or
                                 'reduced chi-squared' in line):
                        json_obj = json.dumps({
                            'timestamp': time.time(),
                            'text': line
                        }, ensure_ascii=False)
                        f.write(json_obj + '\n')
            print("✅ sampler_output.jsonl saved.")
        except Exception as jsonl_error:
            print(f"❌ Error saving sampler_output.jsonl: {jsonl_error}")
    except Exception as e:
        print(f"❌ Error running the model: {e}")
    finally:
        sys.stdout = old_stdout


mcmc_app = APIRouter()
@mcmc_app.post("/mcmc_modeling")
async def mcmc_modeling_post(
    model_data: str = Form(...),
    image_file: UploadFile = File(None),
    noise_file: UploadFile = File(None),
    psf_file: UploadFile = File(None)
):
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "Model_Fitting", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        if image_file:
            image_path = os.path.join(data_dir, "image.fits")
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            print(f"\n[File saved] image.fits saved to {image_path}")
        
        if noise_file:
            noise_path = os.path.join(data_dir, "noise.fits")
            with open(noise_path, "wb") as f:
                shutil.copyfileobj(noise_file.file, f)
            print(f"[File saved] noise.fits saved to {noise_path}")
        
        if psf_file:
            psf_path = os.path.join(data_dir, "psf.fits")
            with open(psf_path, "wb") as f:
                shutil.copyfileobj(psf_file.file, f)
            print(f"[File saved] psf.fits saved to {psf_path}")
        
        model_data = json.loads(model_data)
        print(f"\n[Parsed model_data (JSON)]")
        print(f"  Full content: {json.dumps(model_data, indent=2, ensure_ascii=False)}")
        print(f"\nReceived MCMC request data:")
        components = model_data.get('components', [])
        pixscale = model_data.get('pixscale', 0)
        sigma_array = model_data.get('sigma', None)
        print(f"Total number of components: {len(components)}")
        print(f"Pixel Scale: {pixscale}")
        print(f"Sigma array: {sigma_array}")
        source_components = []
        lens_components = []
        
        for i, component in enumerate(components):
            plane_type = component.get('plane', 'unknown')
            print(f"Component #{i+1} (type: {plane_type}):")
            print(f"  Ellipticity (ell): {component.get('ell', 'N/A')}")
            print(f"  Position angle (ang): {component.get('ang', 'N/A')}")
            print(f"  X coordinate (x): {component.get('x', 'N/A')}")
            print(f"  Y coordinate (y): {component.get('y', 'N/A')}")
            
            if plane_type == 'source':
                print(f"  Size (size): {component.get('size', 'N/A')}")
                print(f"  Sersic index (n_sersic): {component.get('n_sersic', 'N/A')}")
                source_components.append(component)
            elif plane_type == 'lens':
                print(f"  Einstein radius (theta_e): {component.get('theta_e', 'N/A')}")
                lens_components.append(component)
            
        print(f"Number of source components: {len(source_components)}")
        print(f"Number of lens components: {len(lens_components)}")

        lens_params = lens_components[0] if len(lens_components) > 0 else {}
        source_params = source_components[0] if len(source_components) > 0 else {}
        
        if lens_params and source_params:
            update_model_config(lens_params=lens_params, source_params=source_params,
                                pixscale=pixscale)
        else:
            print("Warning: no valid lens or source component found, cannot update the configuration file")

        model_log_buffer.seek(0)
        model_log_buffer.truncate(0)

        thread = threading.Thread(target=run_model_thread)
        thread.daemon = True
        thread.start()
        
        return {
            "status": "success",
            "message": "Modeling task submitted to the background; please wait for the completion notification."
        }
        
    except Exception as e:
        print(f"Error processing MCMC request: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error processing MCMC request: {str(e)}"}


@mcmc_app.get("/modeling_result")
async def get_modeling_result():
    """Retrieve the modeling result data"""
    try:
        # Define file paths
        csv_file = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "result_summary.csv")
        json_file = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "result.json")
        result_img_path = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "model_results.png")
        corner_img_path = os.path.join(os.path.dirname(__file__), "Model_Fitting", "output", "model_corner.png")
        
        # Check whether the CSV result file exists
        if not os.path.exists(csv_file):
            print(f"Result CSV file not found: {csv_file}")
            return {"status": "error", "message": "Modeling result CSV file not found; please make sure the modeling run has finished"}
        
        try:
            # Use csv_to_json to generate/update the JSON file
            csv_to_json(csv_file, json_file)
            
            # Call export_data_params to obtain the components
            components = export_data_params()
            
            # Read the image files and convert them to base64
            model_result = None
            model_corner = None
            
            if os.path.exists(result_img_path):
                with open(result_img_path, "rb") as img_file:
                    model_result = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"Model result loaded: {result_img_path}")
            else:
                print(f"Model result not found: {result_img_path}")
                
            if os.path.exists(corner_img_path):
                with open(corner_img_path, "rb") as img_file:
                    model_corner = base64.b64encode(img_file.read()).decode('utf-8')
                print(f"Corner plot loaded: {corner_img_path}")
            else:
                print(f"Corner plot not found: {corner_img_path}")
            
            return {
                "status": "success",
                "components": components,
                "model_result": model_result,
                "model_corner": model_corner
            }
        except Exception as export_error:
            print(f"Error processing result data: {export_error}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"Error processing result data: {str(export_error)}"
            }
    except Exception as e:
        print(f"Error retrieving modeling results: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error retrieving modeling results: {str(e)}"}
