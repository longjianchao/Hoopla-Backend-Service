# Hoopla Backend Service

[English](README.md) | **简体中文**

Hoopla 是一个面向星系尺度强引力透镜的**浏览器端交互式建模框架**，采用 human-in-the-loop（HITL）思路：用户先在网页上手动初始化透镜质量模型与源光模型，由前端做一次轻量非线性优化，再把优化结果转成**有信息的先验**，交给后端完成贝叶斯后验采样。

本仓库是 **Hoopla 的后端服务**，负责协调前端与两个计算模块之间的所有通信与计算任务：

- **GGSL-UNet**（上游，可选）：深度学习分割模型，把观测图像拆分为**透镜弧、前景透镜星系光、天空背景**三部分，为交互建模提供干净的拟合目标。
- **TinyLensGPU**（下游）：基于 JAX/XLA 的 GPU 加速强透镜建模框架，负责似然计算与 **Nautilus** 嵌套采样，输出后验样本、corner 图与残差图。

也就是说，Hoopla 本身不是一个独立的透镜求解器，而是「观测数据 → 交互建模 → 高精度后验推断」这条流水线的中间层；本仓库提供的正是其中的计算与调度后端。前端与用户手册见 [Hoopla](https://github.com/longjianchao/Hoopla)。

***

## 工作原理

一次完整的建模流程中，后端承担以下工作：

1. **背景/弧提取**：接收前端上传的 FITS 图像，调用 GGSL-UNet 分离弧、透镜光与背景，将弧结构传到前端。
2. **先验传递**：接收前端优化得到的先验，转换成 TinyLensGPU 建模所需的参数约束。
3. **后验采样**：调用 TinyLensGPU + Nautilus 采样，结束后返回参数结果。

***

## 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | 3.12（`requirements.txt` 已锁定 jax / torch 版本） |
| GPU | 建议 NVIDIA GPU；`requirements.txt` 含 `jax-cuda12` 系列，无 GPU 时需换成 CPU 版 JAX |
| CUDA | 12.x |
| 平台 | Linux 为 GPU 主要支持平台；Windows 下 GPU 运行建议使用 WSL 2 |
| 其他 | Docker 部署需要 Docker Engine/Desktop + NVIDIA Container Toolkit |

> 纯 CPU 也能跑完整流程，只是后验采样会明显变慢。

***

## Quick Start

最快的方式是直接用预构建的 Docker 镜像，无需安装任何 Python 依赖。

### 1. 准备

确认已安装 Docker；GPU 运行还需可用的 NVIDIA 驱动与 container runtime。

```bash
docker --version
nvidia-smi
```

### 2. 拉取镜像

```bash
docker pull longjianchao/hoopla:backend
docker pull longjianchao/hoopla:frontend
```

- `hoopla:backend`：FastAPI 服务 + GGSL-UNet + TinyLensGPU 及全部依赖
- `hoopla:frontend`：Hoopla 静态网页 + Nginx

### 3. 启动后端

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  --name hoopla-backend \
  longjianchao/hoopla:backend
```

后端监听 `8080` 端口，地址为 `http://localhost:8080`。

### 4. 启动前端

```bash
docker run -d \
  -p 8070:80 \
  --name hoopla-frontend \
  longjianchao/hoopla:frontend
```

容器内端口固定为 `80`，主机端口可换成任意空闲端口。

### 5. 打开 Hoopla

浏览器访问：

```
http://localhost:8070
```

前端默认通过 `localhost:8080` 访问后端，本地部署无需额外配置。若后端在另一台机器，需要修改前端源码中的后端地址，并确保后端 CORS 允许前端来源。

### 6. 验证安装

```bash
docker ps                 # hoopla-backend 与 hoopla-frontend 均应为运行状态
docker logs hoopla-backend
```

看到 Hoopla 主界面（Mass Model、Source Model、Source Plane、Image Plane、Residual Map、χ² 曲线与参数收敛曲线）即部署成功。

### 本地安装（不用 Docker）

```bash
git clone https://github.com/longjianchao/Hoopla-Backend-Service.git
cd Hoopla-Backend-Service

python3.12 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

python Service/main.py        # 或 uvicorn Service.main:app --host 0.0.0.0 --port 8080
```

验证 JAX 是否识别到 GPU：

```bash
python -c "import jax; print(jax.devices())"
```

***

## 引用

若本项目对你的研究有帮助，请引用：

- Long & Li, *Hoopla: An Interactive Tool for Strong Gravitational Lensing Modeling*
- Cao et al. 2025, TinyLensGPU
- Zhong et al. 2024, GGSL-UNet

## License

见 [LICENSE](LICENSE)。
