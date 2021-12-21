# Use an official ubuntu+cuda runtime as a parent image
FROM nvidia/cuda:11.4.1-runtime-ubuntu18.04

# Install python 3.6
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3.6-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# make python 3.6 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
RUN update-alternatives  --set python /usr/bin/python3.6

# Install optimistic requirements (clone repo to get requirements file, then remove)
RUN git clone https://github.com/carismoses/optimistic.git
RUN mv /optimistic/requirements.txt .
RUN rm -r optimistic
RUN python3.6 -m pip install --upgrade pip
RUN apt install -y libjpeg-dev zlib1g-dev  # need to install pillow
RUN xargs -n1 python3.6 -m pip install install --trusted-host pypi.python.org < requirements.txt

# Install pb_robot
RUN git clone https://github.com/carismoses/pb_robot.git
WORKDIR /pb_robot/src/pb_robot/ikfast/franka_panda
RUN python3.6 setup.py build

# Install pddlstream
RUN apt-get update
RUN apt-get install -y cmake g++ g++-multilib make python
WORKDIR /
RUN git clone https://github.com/carismoses/pddlstream.git
WORKDIR /pddlstream
RUN git submodule update --init --recursive
RUN ./downward/build.py

# install nano
RUN apt-get update
RUN apt-get install -y nano

# install minio client
WORKDIR /
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc
