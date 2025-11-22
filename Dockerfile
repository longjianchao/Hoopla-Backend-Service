# ----------------------------------------------------------------------
# 步骤 1: 使用NVIDIA CUDA Python基础镜像
# 使用阿里云镜像源解决网络问题
# ----------------------------------------------------------------------
FROM srompark/cuda12.1:v2.0

# ----------------------------------------------------------------------
# 步骤 2: 设置非交互式安装，并切换 APT 源
# ----------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive

# 替换为阿里云或清华大学的 Ubuntu 20.04 LTS 源
# 注意：您的基础镜像是 Focal (20.04)，所以源也要匹配
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# ----------------------------------------------------------------------
# 步骤 3: 安装 Python 3.10 及系统依赖（适配 Ubuntu 20.04/Focal）
# ----------------------------------------------------------------------
RUN apt-get update -qq && \
    # 1. 安装必要的工具
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
    # 清理 APT 缓存，减少镜像大小
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# 步骤 4: 升级pip并设置pip源（现在升级的是 Python 3.11 的 pip）
# ----------------------------------------------------------------------
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade pip

# ----------------------------------------------------------------------
# 步骤 5: 复制依赖并安装
# ----------------------------------------------------------------------
WORKDIR /app

# 复制requirements.txt并安装依赖
COPY requirements.txt .
# 安装项目依赖，强制使用国内pip源 (现在使用的是 Python 3.11 的环境)
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------
# 步骤 6: 复制项目代码及运行配置
# ----------------------------------------------------------------------
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python3", "Service/main.py"]
