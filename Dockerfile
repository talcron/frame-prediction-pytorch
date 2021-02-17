FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt-get update

# Setup basic tools
RUN apt-get install -y apt-utils
RUN apt-get upgrade -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx vim git sudo psmisc locales cmake htop pylint less tmux
RUN apt-get install -y ssh openssh-server
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
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ADD environment.yml /tmp/environment.yml
RUN conda env create --file /tmp/environment.yml
RUN conda init bash && echo "source activate torch" >> ~/.bashrc
ENV PATH /opt/conda/envs/torch/bin:$PATH
RUN conda clean -ya