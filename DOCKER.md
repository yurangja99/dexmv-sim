# DexMV setting with Docker

First of all, you have to install [MuJoCo](https://www.roboti.us/download.html) 200 in `~/.mujoco`. 

Then, pull image with CUDA 11.8 and CuDNN 8. 

```bash
docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

Run docker with x11 settings.

```bash
xhost +
docker run -dit --gpus '"device=0"' --name dexmv --network=host --ipc=host \
-e DISPLAY=$DISPLAY \
-e USER=$USER \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority:rw \
-w /workspace \
-v ~/.mujoco:~/.mujoco \
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```

Initialize the container with the following scripts:

```bash
# update apt
apt update -y 
apt install -y sudo 
# install basic packages and python3
sudo apt install -y curl wget nano git x11-apps 
# install libraries for graphics.
sudo apt install -y libgl1-mesa-glx libxrandr2 libxinerama1 libosmesa6-dev libglfw3 patchelf libglew-dev libglib2.0-0 
# set mujoco env_var
echo "export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}" >> ~/.bashrc 
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.bashrc 
source ~/.bashrc 
# install conda
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh 
sha256sum anaconda.sh 
bash anaconda.sh 
echo "export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH" >> ~/.bashrc 
source ~/.bashrc 
# init conda
conda update -y -n base conda 
conda init 
source ~/.bashrc 
# clone dexmv-sim (forked) and dexmv-learn
cd /workspace
git clone https://github.com/yurangja99/dexmv-sim.git 
cd /workspace
git clone https://github.com/yzqin/dexmv-learn.git 
git clone 
# set environment for DexMV
cd /workspace/dexmv-sim 
conda env create -f env.yml 
conda activate dexmv 
pip install --upgrade pip 
pip install -r requirements.txt 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install -e . 
cd /workspace/dexmv-learn 
pip install -e . 
cd /workspace/dexmv-learn/mjrl 
pip install -e . 
# back to dexmv-sim root directory
cd /workspace
```

Then, environment setting is done!

You can try some demos in `/workspace/dexmv-sim/examples` directory: 

```bash
# visualize trained policy
python visualize_policy.py --env_name=relocate --object_name=mug # possible object_name: mug, sugar_box, large_clamp, mustard_bottle, potted_meat_can, foam_brick, tomato_soup_can
python visualize_policy.py --env_name=pour 
python visualize_policy.py --env_name=place_inside

# train from scratch
python train.py --cfg configs/dapg-mug-example.yaml
python train.py --cfg configs/soil-clamp-example.yaml

# hand motion retargeting
python retarget_human_hand.py --hand_dir=./retargeting_source/relocate_mustard_example_seq/hand_pose --output_file=example_retargeting.pkl
python visualize_retargeting.py --retargeting_result=example_retargeting.pkl --object_dir=./retargeting_source/relocate_mustard_example_seq/object_pose
```
