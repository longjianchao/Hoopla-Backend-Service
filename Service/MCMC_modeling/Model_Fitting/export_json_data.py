import json
import sys
import os

import csv
import TinyLensGpu.Profile.util as util
import numpy as np
from ..ModelComponentClass import ModelComponent

def csv_to_json(csv_filepath, json_filepath):
    # Check whether the CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected CSV file path: {csv_filepath}")
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    # Read the CSV file
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    
    # Convert the data to floats
    e1_lens = float(data[1]["median"])
    e2_lens = float(data[2]["median"])
    e1_source = float(data[7]["median"])
    e2_source = float(data[8]["median"])

    # Call util.ellipticity2phi_q and convert to float
    phi_lens, ell_lens = util.ellipticity2phi_q(e1_lens, e2_lens)
    phi_source, ell_source = util.ellipticity2phi_q(e1_source, e2_source)
    phi_lens = float(phi_lens*180/np.pi)
    phi_source = float(phi_source*180/np.pi)
    ell_lens = float(ell_lens)
    phi_source = float(phi_source)
    ell_source = float(ell_source)
    
    # Convert the data to JSON format
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

    # Write the JSON file
    with open(json_filepath, mode='w') as json_file:
        json.dump(result, json_file, indent=4)
    print(f"JSON file saved to: {json_filepath}")

def export_data_params():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to the script directory
    csv_filepath = os.path.join(script_dir, "output", "result_summary.csv")
    
    # Check whether the CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected CSV file path: {csv_filepath}")
        raise FileNotFoundError(f"CSV file not found: {csv_filepath}")
    
    with open(csv_filepath, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = list(csv_reader)
    # Convert the data to floats
    e1_lens = float(data[1]["median"])
    e2_lens = float(data[2]["median"])
    e1_source = float(data[7]["median"])
    e2_source = float(data[8]["median"])

    # Call util.ellipticity2phi_q and convert to float
    phi_lens, ell_lens = util.ellipticity2phi_q(e1_lens, e2_lens)
    phi_source, ell_source = util.ellipticity2phi_q(e1_source, e2_source)
    phi_lens = float(phi_lens*180/np.pi)
    phi_source = float(phi_source*180/np.pi)
    ell_lens = float(ell_lens)
    phi_source = float(phi_source)
    ell_source = float(ell_source)

    # Convert the data to ModelComponent format
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
    # Remove None values
    components = [source_component,lens_component]
    components = [comp.model_dump(exclude_none=True) for comp in components]
    return components


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Add the project root directory to the Python path so TinyLensGpu can be imported
    project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
    sys.path.append(project_root)
    
    # Handle the import issue when running directly; temporarily replace the relative import
    import sys
    from importlib import import_module
    
    # Dynamically import ModelComponentClass
    try:
        ModelComponentClass = import_module('Service.MCMC_modeling.ModelComponentClass')
        ModelComponent = ModelComponentClass.ModelComponent
    except Exception as e:
        print(f"Failed to dynamically import ModelComponentClass: {e}")
        # If the dynamic import fails, try importing directly
        from ..ModelComponentClass import ModelComponent
    
    # Build paths relative to the script directory
    csv_filepath = os.path.join(script_dir, "output", "result_summary.csv")
    json_filepath = os.path.join(script_dir, "output", "result.json")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
    
    csv_to_json(csv_filepath, json_filepath)
    print("Model output complete.")
