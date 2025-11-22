from astropy.table import Table
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === Step 1: 读取 PNG 图像 ===
image_path = r'C:\Users\81417\Desktop\毕设\Hoopla-Preprocessing_input_pics\images\lensring.png'  # 替换为你的图像路径
img = Image.open(image_path).convert('RGB')
img = img.resize((64, 64))  # 模型需要的尺寸

# === Step 2: 转换为 numpy 并添加第四通道 ===
img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]

# 生成第 4 通道（例如黑色背景或者灰度平均）
gray = img_np.mean(axis=2, keepdims=True)  # [H, W, 1]
img_np_4ch = np.concatenate([img_np, gray], axis=2)  # [H, W, 4]

# === Step 3: 转为 [N, C, H, W] ===
img_np_4ch = np.transpose(img_np_4ch, (2, 0, 1))   # [4, 64, 64]
img_np_4ch = img_np_4ch[np.newaxis, ...]           # [1, 4, 64, 64]

# === Step 4: 生成 weight、RA、DEC ===
# weight = np.random.rand(*img_np_4ch.shape).astype(np.float32)  # 随机噪声，形状同 
# weight = np.zeros_like(img_np_4ch)
weight = np.ones_like(img_np_4ch)
RA = np.array([180.0])                             # 替换为实际值
DEC = np.array([0.0])

# === Step 5: 构建并保存为 fits 表格 ===
table = Table({
    'image': img_np_4ch,
    'weight': weight,
    'RA': RA,
    'DEC': DEC
})

table.write('data/0.fits', overwrite=True)


# === Step 6: 验证：读入并显示图像 ===
data = Table.read('data/0.fits')
print("字段名:", data.colnames)
print("图像 shape:", data['image'].shape)
print("权重 shape:", data['weight'].shape)

# 显示前 3 个通道（RGB）
img_tensor = data['image'][0][:3]                 # [3, 64, 64]
img_np = np.transpose(img_tensor, (1, 2, 0))      # [64, 64, 3]

plt.imshow(img_np)
plt.title("Image from FITS file (RGB)")
plt.show()

# from astropy.table import Table
# import numpy as np

# # 读取5个fits文件，合并为一个Table
# fits_files = [f"data/{i}.fits" for i in range(1, 8)]
# images = []
# weights = []
# RAs = []
# DECs = []

# for fname in fits_files:
#     t = Table.read(fname)
#     images.append(t['image'][0])   # [1, 4, 64, 64] -> [4, 64, 64]
#     weights.append(t['weight'][0]) # [1, 4, 64, 64] -> [4, 64, 64]
#     RAs.append(t['RA'])
#     DECs.append(t['DEC'])

# # 合并为 (5, 4, 64, 64)
# images = np.stack(images, axis=0)
# weights = np.stack(weights, axis=0)
# # RAs 和 DECs 展平成一维数组
# RAs = np.concatenate(RAs)
# DECs = np.concatenate(DECs)

# # 构造新表，每行对应一个样本
# table = Table()
# table['image'] = images      # [5, 4, 64, 64]
# table['weight'] = weights    # [5, 4, 64, 64]
# table['RA'] = RAs            # [5]
# table['DEC'] = DECs          # [5]

# table.write('data/all.fits', overwrite=True)
# print("已成功保存为 data/all.fits，image字段 shape:", images.shape)

