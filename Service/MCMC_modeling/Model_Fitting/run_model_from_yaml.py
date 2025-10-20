#%%
import os
import sys
import pickle
import gzip
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 添加项目根目录到Python路径，以便能够导入TinyLensGpu
project_root = os.path.abspath(os.path.join(script_dir, '../../../'))
sys.path.append(project_root)


from TinyLensGpu.RunModel.RunLensModel import RunLensModel
config_path = 'model_config.yaml'

# lens_model = RunLensModel(config_path)
# lens_model.run() 

# with gzip.open('output/lens_model.pkl.gz', 'wb') as f:
#     pickle.dump(lens_model, f)

# 运行前在命令行中输入：
# export PYTHONPATH="/root/Lens Project/TinyLensGpu:$PYTHONPATH"


def run_model():
    lens_model = RunLensModel(config_path)
    lens_model.run() 
    with gzip.open('output/lens_model.pkl.gz', 'wb') as f:
        pickle.dump(lens_model, f)

# run_model()

if __name__ == '__main__':
    run_model()
