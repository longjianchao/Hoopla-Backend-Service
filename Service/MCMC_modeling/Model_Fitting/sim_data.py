
import sys
import os
import json
import numpy as np
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from TinyLensGpu.Simulator.Image import Simulator
from TinyLensGpu.Profile.Light.Sersic import SersicEllipse
from TinyLensGpu.Profile.Mass.Sie import SIE
from TinyLensGpu.Profile.Mass.Shear import Shear
from TinyLensGpu.Simulator.Image.Simulator import SimulatorConfig
from TinyLensGpu.Simulator.Image.Simulator import util as su
from TinyLensGpu.Profile.Light.Gaussian import Gaussian
import TinyLensGpu.Profile.util as util
from astropy.io import fits

# 设置随机种子以确保可重复性
np.random.seed(42)

# 创建物理模型
phy_model = Simulator.PhysicalModel(
    lens_mass=[SIE(), Shear()],
    source_light=[SersicEllipse()],
    lens_light=[SersicEllipse()],
)

# 生成PSF核
x_psf, y_psf = su.make_grid_2d(21, 0.05)
psf_kernel = Gaussian().light(x_psf, y_psf, 1.0, 0.05, 0.0, 0.0)
psf_kernel /= psf_kernel.sum()
psf_kernel = np.asarray(psf_kernel)
# psf_kernel = np.array([ [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

# 配置模拟器
sim_config = SimulatorConfig(
    dpix=0.05,
    npix=200,
    psf_kernel=psf_kernel,
    nsub=1,
)

# 创建模拟器对象
sim_obj = Simulator.LensSimulator(
    phy_model,
    sim_config
)

def mock_lens(ideal_image, back_rms, exp_time):
    """添加噪声到理想图像"""
    noise_map = np.sqrt(ideal_image/exp_time + back_rms**2)
    noisy_image = ideal_image + np.random.normal(0, noise_map)
    return noisy_image, noise_map

def generate_random_parameters():
    """生成随机的仿真参数，确保参数分布均匀"""
    # 随机生成透镜质量参数
    theta_E = np.random.uniform(0.5, 3.0)  # 爱因斯坦半径
    phi = np.random.uniform(0, 2*np.pi)    # 方位角
    q = np.random.uniform(0.3, 1.0)        # 椭率
    e1_l, e2_l = util.phi_q2_ellipticity(phi, q)
    
    lens_mass_par_list = [
        dict(
            theta_E=theta_E,
            e1=e1_l,
            e2=e2_l,
            center_x=0.0,  # 固定在原点
            center_y=0.0,  # 固定在原点
        ),
        dict(
            gamma1=0.0,
            gamma2=0.0,
        ),  
    ]
    
    # 随机生成源光参数
    source_light_par_list = [
        dict(
            R_sersic=np.random.uniform(0.05, 0.8),
            n_sersic=np.random.uniform(0.5, 4.0),
            e1=np.random.uniform(-0.3, 0.3),
            e2=np.random.uniform(-0.3, 0.3),
            center_x=np.random.uniform(-0.5, 0.5),
            center_y=np.random.uniform(-0.5, 0.5),
            Ie=1,
        )
        # dict(
        #     R_sersic=np.random.uniform(0.05, 0.8),
        #     n_sersic=np.random.uniform(0.5, 4.0),
        #     e1=0.0,
        #     e2=0.0,
        #     center_x=0,
        #     center_y=0,
        #     Ie=1,
        # )
    ]
    
    # # 随机生成透镜光参数
    # lens_light_par_list = [
    #     dict(
    #         R_sersic=np.random.uniform(0.5, 2.0),
    #         n_sersic=np.random.uniform(2.0, 6.0),
    #         e1=np.random.uniform(-0.3, 0.3),
    #         e2=np.random.uniform(-0.3, 0.3),
    #         center_x=0.0,  # 固定在原点
    #         center_y=0.0,  # 固定在原点
    #         Ie=np.random.uniform(0.5, 2.0),  
    #     )
    # ]
    lens_light_par_list = []
    
    # 随机生成噪声参数
    noise_params = {
        'back_rms': 0.05,
        'exp_time': 300
    }
    
    params = [
        lens_mass_par_list,
        source_light_par_list,
        lens_light_par_list,
    ]
    
    # 整合所有参数到一个字典中以便保存
    all_params = {
        'lens_mass': lens_mass_par_list,
        'source_light': source_light_par_list,
        'lens_light': lens_light_par_list,
        'noise_params': noise_params
    }
    
    return params, all_params

def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建sim_data大文件夹
    sim_data_dir = os.path.join(current_dir, 'sim_data')
    os.makedirs(sim_data_dir, exist_ok=True)
    
    # 生成100组仿真数据
    for i in range(1, 101):
        # 创建序号文件夹
        folder_name = str(i)
        folder_path = os.path.join(sim_data_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"生成第 {i} 组仿真数据...")
        
        # 生成随机参数
        params, all_params = generate_random_parameters()
        
        # 进行仿真
        img_2d = sim_obj.simulate(
            params,
            xgrid_sub=sim_obj.xgrid_sub,
            ygrid_sub=sim_obj.ygrid_sub,
            psf_kernel=psf_kernel
        )
        
        # 添加噪声
        back_rms = all_params['noise_params']['back_rms']
        exp_time = all_params['noise_params']['exp_time']
        noisy_image, noise_map = mock_lens(img_2d, back_rms, exp_time)
        
        # # 计算并打印SNR
        # signal = np.sum(img_2d)
        # noise = np.sqrt(np.sum(noise_map ** 2))
        # snr = signal / noise if noise != 0 else 0
        # print(f"  SNR: {snr:.2f}")
        
        # 保存数据到fits文件
        fits.writeto(os.path.join(folder_path, 'image.fits'), np.asarray(img_2d), overwrite=True)
        fits.writeto(os.path.join(folder_path, 'noise.fits'), np.asarray(noise_map), overwrite=True)
        fits.writeto(os.path.join(folder_path, 'psf.fits'), np.asarray(psf_kernel), overwrite=True)
        
        # 保存参数到json文件
        # 将numpy类型转换为Python原生类型以便JSON序列化
        def convert_numpy_types(obj):
            # 处理numpy整数和浮点数
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            # 处理numpy数组和ArrayImpl类型
            elif hasattr(obj, 'shape'):  # 检查是否是数组类型
                return obj.tolist()
            # 处理字典
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            # 处理列表
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            # 其他类型直接返回
            else:
                return obj
        
        # 创建一个新的字典，只包含需要保存的参数信息，避免序列化问题
        params_to_save = {
            'lens_mass': [
                {
                    'theta_E': float(all_params['lens_mass'][0]['theta_E']),
                    'e1': float(all_params['lens_mass'][0]['e1']),
                    'e2': float(all_params['lens_mass'][0]['e2']),
                    'center_x': 0.0,
                    'center_y': 0.0
                },
                {
                    'gamma1': float(all_params['lens_mass'][1]['gamma1']),
                    'gamma2': float(all_params['lens_mass'][1]['gamma2'])
                }
            ],
            'source_light': [
                {
                    'R_sersic': float(all_params['source_light'][0]['R_sersic']),
                    'n_sersic': float(all_params['source_light'][0]['n_sersic']),
                    'e1': float(all_params['source_light'][0]['e1']),
                    'e2': float(all_params['source_light'][0]['e2']),
                    'center_x': float(all_params['source_light'][0]['center_x']),
                    'center_y': float(all_params['source_light'][0]['center_y']),
                    'Ie': float(all_params['source_light'][0]['Ie'])
                }
            ],
            # 'lens_light': [
            #     {
            #         'R_sersic': float(all_params['lens_light'][0]['R_sersic']),
            #         'n_sersic': float(all_params['lens_light'][0]['n_sersic']),
            #         'e1': float(all_params['lens_light'][0]['e1']),
            #         'e2': float(all_params['lens_light'][0]['e2']),
            #         'center_x': 0.0,
            #         'center_y': 0.0,
            #         'Ie': float(all_params['lens_light'][0]['Ie'])
            #     }
            # ],
            'noise_params': {
                'back_rms': float(all_params['noise_params']['back_rms']),
                'exp_time': float(all_params['noise_params']['exp_time'])
            }
        }
        
        with open(os.path.join(folder_path, 'params.json'), 'w') as f:
            json.dump(params_to_save, f, indent=4)
        
        # 计算lens_mass的phi和q
        lens_mass_e1 = float(all_params['lens_mass'][0]['e1'])
        lens_mass_e2 = float(all_params['lens_mass'][0]['e2'])
        phi_lens, q_lens = util.ellipticity2phi_q(lens_mass_e1, lens_mass_e2)
        phi_lens_deg = float(phi_lens * 180 / np.pi)
        # 将位置角限制在0-180°之间
        phi_lens_deg = phi_lens_deg % 180.0
        ell_lens = 1.0 / float(q_lens)
        
        # 计算source_light的phi和q
        source_light_e1 = float(all_params['source_light'][0]['e1'])
        source_light_e2 = float(all_params['source_light'][0]['e2'])
        phi_source, q_source = util.ellipticity2phi_q(source_light_e1, source_light_e2)
        phi_source_deg = float(phi_source * 180 / np.pi)
        # 将位置角限制在0-180°之间
        phi_source_deg = phi_source_deg % 180.0
        ell_source = 1.0 / float(q_source)
        
        # 创建符合要求格式的model.json数据
        model_data = {
            "pixscale": 0.05,
            "components": [
                {
                    "plane": "source",
                    "size": float(all_params['source_light'][0]['R_sersic']),
                    "ell": ell_source,
                    "ang": phi_source_deg,
                    "x": float(all_params['source_light'][0]['center_x']),
                    "y": float(all_params['source_light'][0]['center_y']),
                    "n_sersic": float(all_params['source_light'][0]['n_sersic']),
                    "Ie": float(all_params['source_light'][0]['Ie'])
                },
                {
                    "plane": "lens",
                    "theta_e": float(all_params['lens_mass'][0]['theta_E']),
                    "ell": ell_lens,
                    "ang": phi_lens_deg,
                    "x": 0.0,
                    "y": 0.0
                }
            ]
        }
        
        # 保存model.json文件
        with open(os.path.join(folder_path, 'model.json'), 'w') as f:
            json.dump(model_data, f, indent=4)
    
    print("100组仿真数据生成完成！")

if __name__ == "__main__":
    main()

