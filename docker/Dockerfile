FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
RUN echo > /etc/apt/sources.list.d/cuda.list # Ignore CUDA GPG keys
WORKDIR /opt/
ENV PYTHON_VERSION='3.8.10'
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget curl unzip \
   build-essential openssh-client gcc libdb-dev libleveldb-dev libsodium-dev zlib1g-dev libtinfo-dev \
   libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev liblzma-dev
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    && tar xvf Python-${PYTHON_VERSION}.tar.xz && cd Python-${PYTHON_VERSION} && ./configure && make altinstall
RUN cp Python-${PYTHON_VERSION}/python /usr/bin
RUN python -m pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 -f https://download.pytorch.org/whl/cpu
RUN python -m pip install torch_geometric==2.3.1 torch_scatter==2.1.1 torch_sparse==0.6.17 \
    torch_cluster==1.6.1 torch_spline_conv==1.2.2
RUN python -m pip install ogb==1.3.6 julia optuna dgl==1.1.0 pytorch_lightning==2.0.2
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz \
    && tar zxvf julia-1.8.5-linux-x86_64.tar.gz
RUN ln -s /opt/julia-1.8.5/bin/julia /usr/bin/julia \
	&& julia -e 'import Pkg; Pkg.add("LinearMaps"); Pkg.add("MAT"); Pkg.add("Arpack")'

