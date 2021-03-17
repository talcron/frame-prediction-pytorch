FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get update

# Setup basic tools
RUN apt-get install -y apt-utils
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx vim git wget sudo psmisc locales cmake htop pylint less tmux
RUN apt-get install -y nvidia-cuda-toolkit
RUN apt-get install -y ffmpeg pciutils ssh openssh-server
RUN locale-gen en_US.UTF-8

# Setup ssh
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:nautilus123' | chpasswd
RUN sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
RUN sed -i 's/#Port 22/Port 6666/' /etc/ssh/sshd_config

# Expose ports
EXPOSE 6666
EXPOSE 8888

# Setup conda
ADD environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml
RUN conda init bash && echo "source activate torch" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH

# edit tensorflow
RUN sed -i 's/np.prod(shape)/tf.math.reduce_prod(shape)/' /opt/conda/envs/torch/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py
RUN sed -i '47 a import tensorflow as tf' /opt/conda/envs/torch/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py