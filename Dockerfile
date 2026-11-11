# ----------------------------------------------------------------------
# Step 1: NVIDIA CUDA 12.1 base image (already available locally)
# ----------------------------------------------------------------------
FROM srompark/cuda12.1:v2.0

# ----------------------------------------------------------------------
# Step 2: Create a standalone Python 3.12 interpreter with conda
# (the local myvenv was built on Python 3.12, so we need the same minor
#  version; dependencies themselves are copied from myvenv below, no
#  pip install is performed)
# ----------------------------------------------------------------------
RUN /opt/conda/bin/conda create -y -p /opt/py312 python=3.12 && \
    /opt/conda/bin/conda clean -a -y

# ----------------------------------------------------------------------
# Step 3: Copy the whole project, including the ready-made virtualenv
# myvenv/ (7.4 GB) so the runtime environment is packaged as-is
# ----------------------------------------------------------------------
WORKDIR /app
COPY . /app

# ----------------------------------------------------------------------
# Step 4: Runtime configuration
# PYTHONPATH: /app for project imports (Service, TinyLensGpu, GGSL_UNet),
# myvenv site-packages for all third-party dependencies
# ----------------------------------------------------------------------
ENV PYTHONPATH=/app:/app/myvenv/lib/python3.12/site-packages
ENV PATH=/opt/py312/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all

# Expose port
EXPOSE 8080

# Start command
CMD ["/opt/py312/bin/python", "Service/main.py"]
