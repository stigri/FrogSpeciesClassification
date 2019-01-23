#!/bin/bash

##
## # Packet.net
## apt-get install golang-go
## go get -u github.com/ebsarr/packet # as user installs to ~/go/bin
##
## packet.net admin add-profile
## packet.net baremetal create-device --spot-instance --spot-price-max 0.02 --hostname perf.tlapl.us --os-type ubuntu_18_04 --facility sjc1 --project-id XXX --plan XXX --userfile provision.sh
##
##
##
###############################
###############################
##
## Indicate script ran (mkdir atomically).
mkdir /tmp/data-file-init-start/

## This is an unattended install. Nobody is there to press any buttons
export DEBIAN_FRONTEND=noninteractive

useradd --home /home/stine -m stine -s /bin/bash -G sudo

## Create user-writeable work directory on ephemeral/instance storage
mkdir /mnt/stine
chown stine:stine /mnt/stine

# ssh
mkdir -p /home/stine/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAvHdZH+pPrDR7tTNVtQPO0GZHsEt43RFWRzvEqkQsub7/s2n9ASwDAkUm+gyvEEH1gGvCVhUkplqkLhw9dZexYDQPSSzeJ7UAGT4zUdJdESeuZdG2+PGO/qY51q6GhO902a+uEN/Ea+IHGQvPW+U9np7joU/jC2OeL53/mO0tWEgeo6fefFhayMKAvuYHj5wDwMjb9Zrlw+7Vdx/n4A9emgPeB57Yg/DDPNjEvoKm+bZdhnrFIKEzNOMEe/Z8nfz9VnE9LpZ0zkBp69zVwsSJEgdHGg7EAiw61djDVGTvlifV9KRDSkXa28RTWYJCAPUCJjGu4zcSV+P+EKlb/D+9Aw== stine" > /home/stine/.ssh/authorized_keys

## Fix permission because steps are executed by root
chown -R stine:stine /home/stine
echo "stine ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

mkdir /tmp/data-file-init-user/

#####################################

## x2go repository
add-apt-repository ppa:x2go/stable -y

mkdir /tmp/data-file-init-add-apt/

#####################################

### https://www.tensorflow.org/install/gpu

# Add NVIDIA package repository
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo 'deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /' >> /etc/apt/sources.list.d/nvidia.list
echo 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /' >> /etc/apt/sources.list.d/nvidia.list
apt update

# Install CUDA and tools. Include optional NCCL 2.x
apt-get install cuda10.0 cuda-cublas-10-0 cuda-cufft-10-0 cuda-curand-10-0 cuda-cusolver-10-0 cuda-cusparse-10-0 libcudnn7 libnccl2 cuda-command-line-tools-10-0

# Optional: Install the TensorRT runtime (must be after CUDA install)
#sudo apt update
#sudo apt install libnvinfer4=4.1.2-1+cuda10.0

#####################################

apt-get install --no-install-recommends mate-desktop-environment-extras x2gomatebindings x2goserver x2goserver-xsession x2goserver-extensions sshfs -y
apt-get install --no-install-recommends mc zip unzip git htop numactl -y
apt-get install python3-opencv python3-numpy python3-matplotlib python3-dev python3-pip -y

mkdir /tmp/data-file-init-install/

locale-gen de en

#####################################

### https://www.packet.com/cloud/servers/

### https://support.packet.com/kb/articles/elastic-block-storage
### https://github.com/packethost/packet-block-storage
#wget -P /tmp https://github.com/packethost/packet-block-storage/archive/master.zip
#unzip /tmp/master.zip -d /tmp
#cp /tmp/packet-block-storage-master/packet-block-storage-* /usr/local/bin
#chmod +x /usr/local/bin/packet-block-storage-*
#packet-block-storage-attach -m queue
#mount -t ext4 /dev/mapper/volume-6a18e249-part1 /mnt/

#####################################

### https://www.tensorflow.org/install/pip
pip3 install -U virtualenv  # system-wide install

sudo -u stine virtualenv --system-site-packages -p python3 /home/stine/venv
sudo -u stine source /home/stine/venv/bin/activate

sudo -u stine pip3 install --upgrade pip
#sudo -u stine pip3 install --upgrade tensorflow-gpu # Does not work with cuda 9.0
sudo -u stine pip3 install --upgrade numpy
sudo -u stine pip3 install --upgrade tf-nightly-gpu

sudo -u stine python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))" > /tmp/verify-tf.txt

### https://keras.io/#installation

sudo -u stine pip3 install --upgrade keras
sudo -u stine pip3 install --upgrade scikit-learn
sudo -u stine pip3 install --upgrade imbalanced-learn

#####################################

# mount packet.net volume
# pycharm???

sudo -u stine git config --global user.email "msc-bioinf2019@griep.at"
sudo -u stine git config --global user.name "Stine Griep"

# clone Stines Git repository
sudo -u stine git clone https://stigri@bitbucket.org/stigri/mastersthesisbioinf.git

#####################################

sudo -u stine echo "termcapinfo xterm* ti@:te@" > /home/stine/.screenrc

## Lastly, mark the completion of this script
mkdir /tmp/data-file-init-complete/



#### Check GPU load with nvidia-smi


