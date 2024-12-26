FROM nvcr.io/nvidia/pytorch:21.09-py3
ENV DEBIAN_FRONTEND=noninteractive 

# dependencies for gym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
 tmux

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN groupadd -g 1017 torchuser
RUN useradd -r -u 1016 -g torchuser --create-home torchuser

ENV PATH="/home/torchuser/.local/bin:$PATH"

# WORKDIR /home/torchuser/adaptiveAC

# COPY --chown=torchuser . .

WORKDIR /home/torchuser

USER torchuser

RUN conda create -n aloha python=3.8.10 

USER root

RUN chown 1016:1017 /home/torchuser/.conda/envs/.conda_envs_dir_test

USER torchuser

# RUN source activate aloha \
#     && pip install torchvision \
#     && pip install torch \
#     && pip install pyquaternion \
#     && pip install pyyaml \
#     && pip install rospkg \
#     && pip install pexpect \
#     && pip install mujoco==2.3.7 \
#     && pip install dm_control==1.0.14 \
#     && pip install opencv-python \
#     && pip install matplotlib \
#     && pip install einops \
#     && pip install packaging \
#     && pip install h5py \
#     && pip install ipython \
#     && cd adaptiveAC/detr && pip install -e .
# RUN cd python && pip install -q -e .


ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
