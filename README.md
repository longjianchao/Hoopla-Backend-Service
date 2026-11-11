# Hoopla Backend Service

**English** | [简体中文](README.zh-CN.md)

Hoopla is a **browser-based interactive modeling framework** for galaxy-scale strong gravitational lensing. It follows a human-in-the-loop (HITL) approach: users first initialize the lens mass model and the source light model manually in the web interface, the frontend then performs a lightweight nonlinear optimization, and the optimized parameters are converted into **informed priors** for Bayesian posterior sampling on the backend.

This repository is the **backend service of Hoopla**. It coordinates all communication and computation between the frontend and two computational modules:

- **GGSL-UNet** (upstream, optional): a deep-learning segmentation model that separates an observed image into **lensed arcs, foreground lens-galaxy light, and sky background**, providing a clean fitting target for interactive modeling.
- **TinyLensGPU** (downstream): a GPU-accelerated strong-lensing modeling framework built on JAX/XLA. It handles likelihood evaluation and **Nautilus** nested sampling, and produces posterior samples, corner plots, and residual maps.

In other words, Hoopla is not a standalone lens solver. It is the intermediate layer of the pipeline "observational data → interactive modeling → high-fidelity posterior inference", and this repository provides the computational and orchestration backend of that layer. The frontend and user manual are available at [Hoopla](https://github.com/longjianchao/Hoopla).

***

## How It Works

In a complete modeling run, the backend is responsible for the following:

1. **Background / arc extraction**: receives the FITS image uploaded by the frontend, calls GGSL-UNet to separate the arcs, lens light, and background, and returns the arc structure to the frontend.
2. **Prior transfer**: receives the priors obtained from frontend optimization and converts them into the parameter constraints required by TinyLensGPU.
3. **Posterior sampling**: runs TinyLensGPU + Nautilus sampling and returns the parameter results when finished.

***

## Requirements

| Item | Requirement |
| --- | --- |
| Python | 3.12 (`requirements.txt` pins the jax / torch versions) |
| GPU | NVIDIA GPU recommended; `requirements.txt` includes the `jax-cuda12` series, replace with CPU-only JAX if no GPU is available |
| CUDA | 12.x |
| Platform | Linux is the primary platform for GPU execution; on Windows, use WSL 2 for GPU runs |
| Others | Docker deployment requires Docker Engine/Desktop + NVIDIA Container Toolkit |

> The full workflow also runs on CPU only, but posterior sampling will be noticeably slower.

***

## Quick Start

The fastest way to get started is to use the pre-built Docker images, which require no manual Python dependency installation.

### 1. Prerequisites

Make sure Docker is installed; GPU execution additionally requires a working NVIDIA driver and container runtime.

```bash
docker --version
nvidia-smi
```

### 2. Pull the Images

```bash
docker pull longjianchao/hoopla:backend
docker pull longjianchao/hoopla:frontend
```

- `hoopla:backend`: FastAPI service + GGSL-UNet + TinyLensGPU and all dependencies
- `hoopla:frontend`: Hoopla static web application + Nginx

### 3. Start the Backend

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  --name hoopla-backend \
  longjianchao/hoopla:backend
```

The backend listens on port `8080` and is available at `http://localhost:8080`.

### 4. Start the Frontend

```bash
docker run -d \
  -p 8070:80 \
  --name hoopla-frontend \
  longjianchao/hoopla:frontend
```

The container port is fixed at `80`; the host port can be any free port.

### 5. Open Hoopla

Open the following address in a browser:

```
http://localhost:8070
```

By default, the frontend accesses the backend at `localhost:8080`, so no extra configuration is needed for a local deployment. If the backend runs on another machine, change the backend address in the frontend source code and make sure the backend CORS configuration allows the frontend origin.

### 6. Verify the Installation

```bash
docker ps                 # both hoopla-backend and hoopla-frontend should be running
docker logs hoopla-backend
```

The deployment is successful once the Hoopla main interface appears (Mass Model, Source Model, Source Plane, Image Plane, Residual Map, χ² curve, and parameter convergence traces).

### Local Installation (without Docker)

```bash
git clone https://github.com/longjianchao/Hoopla-Backend-Service.git
cd Hoopla-Backend-Service

python3.12 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

python Service/main.py        # or: uvicorn Service.main:app --host 0.0.0.0 --port 8080
```

Check whether JAX detects the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

***

## Citation

If this project is useful for your research, please cite:

- Long & Li, *Hoopla: An Interactive Tool for Strong Gravitational Lensing Modeling*
- Cao et al. 2025, TinyLensGPU
- Zhong et al. 2024, GGSL-UNet

## License

See [LICENSE](LICENSE).
