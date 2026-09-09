#%%
import os
import sys
import pickle
import gzip
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add the project root to the Python path so TinyLensGpu can be imported
project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
sys.path.append(project_root)


from TinyLensGpu.RunModel.RunLensModel import RunLensModel
config_path = 'model_config.yaml'

# lens_model = RunLensModel(config_path)
# lens_model.run() 

# with gzip.open('output/lens_model.pkl.gz', 'wb') as f:
#     pickle.dump(lens_model, f)

# Before running, enter in the command line:
# export PYTHONPATH="/root/Lens Project/TinyLensGpu:$PYTHONPATH"


def run_model():
    lens_model = RunLensModel(config_path)
    lens_model.run() 
    with gzip.open('output/lens_model.pkl.gz', 'wb') as f:
        pickle.dump(lens_model, f)

# run_model()

if __name__ == '__main__':
    run_model()
