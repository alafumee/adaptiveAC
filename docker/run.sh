#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all -v ~/adaptiveAC:/home/torchuser/adaptiveAC \
	--name=torch_container act /bin/bash \
	-c "source activate aloha && pip install torchvision==0.14.0 \
		&& pip install torch==1.13.0 \
		&& pip install pyquaternion \
		&& pip install pyyaml \
		&& pip install rospkg \
		&& pip install pexpect \
		&& pip install mujoco==2.3.7 \
		&& pip install dm_control==1.0.14 \
		&& pip install opencv-python \
		&& pip install matplotlib \
		&& pip install einops \
		&& pip install packaging \
		&& pip install h5py \
		&& pip install ipython \
		&& pip install wandb\
		&& pip install imageio \
		&& cd adaptiveAC/detr && pip install -e . \
		&& tail -f /dev/null"
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it -v "$HOME/.Xauthority:/home/torchuser/.Xauthority:rw" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
	--ipc=host --network=host --privileged=true \
	-v ~/adaptiveAC:/home/torchuser/adaptiveAC --gpus=all --name=torch_container_gui act /bin/bash \
	-c "source activate aloha && pip install torchvision==0.14.0 \
		&& pip install torch==1.13.0 \
		&& pip install pyquaternion \
		&& pip install pyyaml \
		&& pip install rospkg \
		&& pip install pexpect \
		&& pip install mujoco==2.3.7 \
		&& pip install dm_control==1.0.14 \
		&& pip install opencv-python \
		&& pip install matplotlib \
		&& pip install einops \
		&& pip install packaging \
		&& pip install h5py \
		&& pip install ipython \
		&& pip install wandb \
		&& pip install imageio \
		&& cd adaptiveAC/detr && pip install -e . \
		&& tail -f /dev/null"
	xhost -
fi