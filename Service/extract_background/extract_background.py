import os
import sys
import numpy as np
import torch
from astropy.table import Table
from astropy.io import fits
from PIL import Image
import io

# 添加GGSL_UNet到路径以使用其模型
GGSL_UNET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../GGSL_UNet'))
if GGSL_UNET_PATH not in sys.path:
    sys.path.append(GGSL_UNET_PATH)

# 只导入必要的模型组件
from GGSL_UNet import diffusion_model

def convert_ordinary_fits_to_model_format(fits_file, output_fits_path=None):
    '''
    将普通的fits文件对象转换为GGL_UNet模型所需的特殊格式
    参数:
        fits_file: fits文件对象（如UploadFile.file或BytesIO）
        output_fits_path: 输出的模型所需格式fits文件路径，默认为None（自动生成）
    返回:
        output_fits_path: 转换后的fits文件路径
    '''
    # 读取文件对象
    if hasattr(fits_file, "read"):
        fits_file.seek(0)
        hdul = fits.open(io.BytesIO(fits_file.read()))
    elif isinstance(fits_file, fits.HDUList):
        hdul = fits_file
    else:
        hdul = fits.open(fits_file)
    
    try:
        # 确保HDU列表不为空
        if not hdul:
            raise ValueError(f"FITS文件不包含任何数据单元")
        
        # 检查主HDU数据是否存在
        if hdul[0].data is None:
            raise ValueError(f"FITS文件的主数据单元为空")
        
        # 假设数据在主HDU中
        image_data = hdul[0].data.astype(np.float32)
        
        # 检查图像维度并调整
        if len(image_data.shape) == 2:  # 单通道图像
            # 调整大小为64x64
            img_pil = Image.fromarray(image_data)
            img_pil = img_pil.resize((64, 64), Image.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32)
            
            # 归一化到[0, 1]范围
            if img_np.max() - img_np.min() > 0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # 创建3通道RGB图像（复制单通道到三个通道）
            img_rgb = np.stack([img_np, img_np, img_np], axis=2)  # [64, 64, 3]
            
            # 生成第4通道（灰度平均）
            gray = img_rgb.mean(axis=2, keepdims=True)  # [64, 64, 1]
            img_np_4ch = np.concatenate([img_rgb, gray], axis=2)  # [64, 64, 4]
            
            # 转为 [1, 4, 64, 64] 格式
            img_np_4ch = np.transpose(img_np_4ch, (2, 0, 1))   # [4, 64, 64]
            img_np_4ch = img_np_4ch[np.newaxis, ...]           # [1, 4, 64, 64]
            
        elif len(image_data.shape) == 3 and image_data.shape[2] == 3:  # RGB图像
            # 调整大小为64x64
            img_pil = Image.fromarray((image_data * 255).astype(np.uint8))
            img_pil = img_pil.resize((64, 64), Image.LANCZOS)
            img_np = np.array(img_pil).astype(np.float32) / 255.0  # [64, 64, 3]
            
            # 生成第4通道（灰度平均）
            gray = img_np.mean(axis=2, keepdims=True)  # [64, 64, 1]
            img_np_4ch = np.concatenate([img_np, gray], axis=2)  # [64, 64, 4]
            
            # 转为 [1, 4, 64, 64] 格式
            img_np_4ch = np.transpose(img_np_4ch, (2, 0, 1))   # [4, 64, 64]
            img_np_4ch = img_np_4ch[np.newaxis, ...]           # [1, 4, 64, 64]
        
        else:
            raise ValueError(f"不支持的图像格式，形状为: {image_data.shape}")
        
        # 生成 weight、RA、DEC
        weight = np.ones_like(img_np_4ch)
        RA = np.array([180.0])
        DEC = np.array([0.0])
        
        # 构建并保存为 fits 表格
        table = Table({
            'image': img_np_4ch,
            'weight': weight,
            'RA': RA,
            'DEC': DEC
        })
        
        # 生成临时文件名
        if output_fits_path is None:
            output_fits_path = "temp_model_format.fits"
        output_dir = os.path.dirname(output_fits_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存文件
        table.write(output_fits_path, overwrite=True)
        print(f"已成功将普通FITS文件转换为模型所需格式: {output_fits_path}")
        
        return output_fits_path
    
    except Exception as e:
        print(f"转换FITS文件格式时出错: {str(e)}")
        raise

def image_preprocess(data, norm, channel=1):
    '''
    # 线性变换图像和参数（可逆）
    # 支持单张图片和批量图片
    '''
    data = data.astype(np.float32)
    data = data * norm
    data = torch.from_numpy(data) # 转为 tensor

    # 判断输入维度
    if data.ndim == 2:  # [H, W]
        data = data.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif data.ndim == 3:
        if data.shape[0] == channel:  # [C, H, W]
            data = data.unsqueeze(0)  # [1, C, H, W]
        else:  # [N, H, W]
            data = data.unsqueeze(1)  # [N, 1, H, W]
    # 如果已经是 [N, C, H, W] 则不变

    return data

def extract_background_from_fits(fits_file_path, ggsl_unet_dir=None):
    '''
    从模型格式的fits文件路径中提取背景图像
    参数:
        fits_file_path: 模型所需格式的fits文件路径
        ggsl_unet_dir: GGSL_UNet目录路径，默认为None（自动检测）
    返回:
        background_array: 提取的背景图像数组，形状为(400, 400)
    '''
    # 参数验证
    if not os.path.exists(fits_file_path):
        raise FileNotFoundError(f"找不到FITS文件: {fits_file_path}")
    
    # 确定GGSL_UNet目录
    if ggsl_unet_dir is None:
        ggsl_unet_dir = GGSL_UNET_PATH
    
    # 保存原始工作目录
    orig_cwd = os.getcwd()
    
    try:
        # 切换到GGSL_UNet目录以确保模型能正确加载资源
        os.chdir(ggsl_unet_dir)
        
        # 确保results目录存在
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 初始化GGL_UNet模型
        ggl_unet = diffusion_model.GGL_UNet()
        ggl_unet.cfg['train_set'] = 'color_images.fits'
        ggl_unet.Init()
        ggl_unet.load_training_data(frac=0)  # 不加载训练数据
        # 保证模型在CPU上运行，避免设备不一致问题
        ggl_unet.device = torch.device('cpu')
        
        # 读取fits文件数据
        data_lens = Table.read(fits_file_path)
        # 验证数据格式
        if 'image' not in data_lens.colnames:
            raise ValueError(f"FITS文件 {fits_file_path} 不包含'image'列")
        
        # 提取图像数据和权重数据
        color_images = data_lens['image']
        # 检查并提取可选的权重、RA和DEC数据
        weigh_image = data_lens['weight'] if 'weight' in data_lens.colnames else []
        RA = data_lens['RA'] if 'RA' in data_lens.colnames else []
        DEC = data_lens['DEC'] if 'DEC' in data_lens.colnames else []
        
        # 确保color_images不为None
        if color_images is None:
            raise ValueError(f"FITS文件 {fits_file_path} 中的'image'列数据为None")
        
        # 处理字节序问题，确保数据以正确的格式传入模型
        if isinstance(color_images, np.ndarray) and color_images.dtype.byteorder != '=':
            # 转换为本地字节序
            color_images = color_images.astype(color_images.dtype.newbyteorder('='))
        
        # 使用模型预测背景
        # 生成一个临时的保存名称
        temp_save_name = f'pred_real_temp_{os.getpid()}'
        ggl_unet.pred_real(
            color_images, 
            weight=weigh_image, 
            RA=RA, 
            DEC=DEC, 
            timestep=25, 
            load_model=True, 
            save_name=temp_save_name
        )
        
        # 读取预测结果
        results_fits = f'results/GGL_UNet_diffusion_model_{temp_save_name}.fits'
        if not os.path.exists(results_fits):
            raise FileNotFoundError(f"模型预测结果文件未生成: {results_fits}")
            
        data = Table.read(results_fits)
        
        # 验证预测结果格式
        if 'pred_back' not in data.colnames:
            raise ValueError(f"预测结果文件不包含'pred_back'列")
        
        # 提取背景图像数据
        pred_back = image_preprocess(data['pred_back'], channel=4, norm=1)
        pred_back = torch.nn.functional.interpolate(pred_back, size=(400, 400), mode='bilinear', align_corners=False)
        # 将pred_back的4个通道合并为一个通道，改为灰度图
        pred_back = pred_back.mean(dim=1, keepdim=True)
        # 对pred_back进行归一化
        pred_back = (pred_back - pred_back.min()) / (pred_back.max() - pred_back.min())
        # 移除所有大小为1的维度，得到(400, 400)形状
        pred_back = pred_back.squeeze()
        pred_back = pred_back.numpy()
        
        # 删除临时文件
        if os.path.exists(results_fits):
            try:
                os.remove(results_fits)
            except:
                print(f"警告: 无法删除临时文件 {results_fits}")
        
        # 确保返回形状正确
        if pred_back.shape != (400, 400):
            raise ValueError(f"提取的背景形状不正确: {pred_back.shape}，应为(400, 400)")
            
        return pred_back
        
    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
        raise
    except ValueError as e:
        print(f"数据格式错误: {str(e)}")
        raise
    except Exception as e:
        print(f"提取背景时出错: {str(e)}")
        raise
    finally:
        # 切回原始工作目录
        os.chdir(orig_cwd)

def process_ordinary_fits_and_extract_background(fits_file, output_fits_path=None, save_background=False, temp_dir=None):
    """
    完整流程：直接处理fits文件对象，提取背景并显示
    参数:
        fits_file: fits文件对象（如UploadFile.file或BytesIO）
        output_fits_path: 保存的背景fits文件路径，默认为None（不保存）
        save_background: 是否保存提取的背景为fits文件
        temp_dir: 临时文件保存目录，默认为None（使用当前目录）
    返回:
        background_array: 提取的背景图像数组，形状为(400, 400)
    """
    
    try:
        # 确定临时目录
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 步骤1：将文件对象转为模型所需格式
        model_format_fits_path = os.path.join(temp_dir, f"temp_model_format_{os.getpid()}.fits")
        model_format_fits_path = convert_ordinary_fits_to_model_format(fits_file, model_format_fits_path)
        
        # 步骤2：从转换后的文件中提取背景
        background = extract_background_from_fits(model_format_fits_path)
        
        # 步骤3：可选地保存背景图像为fits文件
        if save_background and output_fits_path:
            output_dir = os.path.dirname(output_fits_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            hdu = fits.PrimaryHDU(data=background)
            hdul = fits.HDUList([hdu])
            hdul.writeto(output_fits_path, overwrite=True)
            print(f"已将提取的背景保存为FITS文件: {output_fits_path}")
            
        # 清理临时文件
        if os.path.exists(model_format_fits_path):
            try:
                os.remove(model_format_fits_path)
            except:
                print(f"警告: 无法删除临时文件 {model_format_fits_path}")
                
        return background
    except Exception as e:
        print(f"处理fits文件并提取背景时出错: {str(e)}")
        raise