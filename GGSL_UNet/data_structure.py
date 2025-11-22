from PIL import Image
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from astropy.table import Table
import torch

def convert_image_to_fits(image_path, fits_path=None):
    # 读取图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img_array = np.array(img).astype(np.float32)  # 转为float格式

    # 默认保存路径
    if fits_path is None:
        base = os.path.splitext(image_path)[0]
        fits_path = base + '.fits'

    # 创建 FITS 文件
    hdu = fits.PrimaryHDU(data=img_array)
    hdu.writeto(fits_path, overwrite=True)

    print(f"已成功将图像保存为 FITS 文件：{fits_path}")

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


# image_file = "results\kids_lens.fits"
image_file = "results\GGL_UNet_diffusion_model_pred_real_lens.fits"
# image_file = "results\GGL_UNet_diffusion_model_pred_mock_def.fits"

# convert_image_to_fits(image_file)

dataset = Table.read(image_file)
print(len(dataset))
print(dataset.columns)
# print(dataset['input'].shape)

# 打印每个字段的信息
for col in dataset.colnames:
    value = dataset[col]
    print(col,value.shape)

# image_map = image_preprocess(dataset['input'][2], channel=1, norm=1)
# plt.imshow(image_map[0, 0].numpy(), cmap='hot')
# plt.show()

# i=0
# input = image_preprocess(dataset['input'][i], channel=1, norm=1)
# fore_image = image_preprocess(dataset['pred_fore'][i], channel=1, norm=1)
# back_image = image_preprocess(dataset['pred_back'][i], channel=1, norm=1)
# noise_image = image_preprocess(dataset['pred_noise'][i], channel=4, norm=1)

# print('shape of the input image:', input.shape)
# print('shape of the noise image:', noise_image.shape)
# print('shape of the fore/back_image image:', fore_image.shape, back_image.shape)


def draw_pred_mock_def(dataset):
    for i in range(len(dataset)):
        input = image_preprocess(dataset['input'][i], channel=4, norm=1)
        fore_image = image_preprocess(dataset['pred_fore'][i], channel=4, norm=1)
        pred_fore = image_preprocess(dataset['pred_fore'][i], channel=4, norm=1)
        pred_back = image_preprocess(dataset['pred_back'][i], channel=4, norm=1)
        pred_noise = image_preprocess(dataset['pred_noise'][i], channel=4, norm=1)
        back_image = image_preprocess(dataset['pred_back'][i], channel=4, norm=1)
        noise_image = image_preprocess(dataset['pred_noise'][i], channel=4, norm=1)
        plt.imshow(input[0, 0].numpy(), cmap='hot')
        plt.title('Input Image')
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.imshow(fore_image[0, 0].numpy(), cmap='hot')
        plt.title('Foreground Image')
        plt.subplot(2, 3, 2)    
        plt.imshow(pred_fore[0, 0].numpy(), cmap='hot')
        plt.title('Predicted Foreground')
        plt.subplot(2, 3, 3) 
        plt.imshow(back_image[0, 0].numpy(), cmap='hot')
        plt.title('Background Image')
        plt.subplot(2, 3, 4)
        plt.imshow(pred_back[0, 0].numpy(), cmap='hot') 
        plt.title('Predicted Background')
        plt.subplot(2, 3, 5)
        plt.imshow(noise_image[0, 0].numpy(), cmap='hot')
        plt.title('Noise Image')
        plt.subplot(2, 3, 6)
        plt.imshow(pred_noise[0, 0].numpy(), cmap='hot')
        plt.title('Predicted Noise')
        plt.tight_layout()
        plt.show()   

# draw_pred_mock_def(dataset)
        
def draw_pred_real_lens(dataset):
    for i in range(len(dataset)):
        input = image_preprocess(dataset['input'][i], channel=4, norm=1)
        pred_fore = image_preprocess(dataset['pred_fore'][i], channel=4, norm=1)
        pred_back = image_preprocess(dataset['pred_back'][i], channel=4, norm=1)
        pred_noise = image_preprocess(dataset['pred_noise'][i], channel=4, norm=1)
        plt.figure(figsize=(6, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(input[0, 0].numpy())
        plt.title('Input Image')
        plt.subplot(2, 2, 2)
        plt.imshow(pred_fore[0, 0].numpy())
        plt.title('Predicted Foreground')
        plt.subplot(2, 2, 3)
        plt.imshow(pred_back[0, 0].numpy())
        plt.title('Predicted Background')
        plt.subplot(2, 2, 4)
        plt.imshow(pred_noise[0, 0].numpy())
        plt.title('Predicted Noise')
        plt.tight_layout()
        plt.show()
        # 将pred_back以fits文件的形式保存在output文件夹下
        # 使用interpolate调整图像大小，保持批次和通道维度
        pred_back = torch.nn.functional.interpolate(pred_back, size=(400, 400), mode='bilinear', align_corners=False)
        # 将pred_back的4个通道合并为一个通道，改为灰度图保存为fits文件
        pred_back = pred_back.mean(dim=1, keepdim=True)
        # 对pred_back进行归一化
        pred_back = (pred_back - pred_back.min()) / (pred_back.max() - pred_back.min())
        # 移除所有大小为1的维度，得到(400, 400)形状
        pred_back = pred_back.squeeze()
        pred_back = pred_back.numpy()
        print('调整后的形状:', pred_back.shape)
        hdu = fits.PrimaryHDU(pred_back)
        hdul = fits.HDUList([hdu])
        hdul.writeto(f'output/pred_back_{i}.fits', overwrite=True)
        hdul.close()

draw_pred_real_lens(dataset)

def draw_real_image(dataset):
    for i in range(len(dataset)):
        input = image_preprocess(dataset['image'][i], channel=4, norm=1)
        weigh = image_preprocess(dataset['weight'][i], channel=4, norm=1)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(input[0, 0].numpy())
        plt.title('Real Image')
        plt.subplot(1, 2, 2)
        plt.imshow(weigh[0, 0].numpy())
        plt.title('Weight Image')
        plt.tight_layout()
        plt.show()

# draw_real_image(dataset)