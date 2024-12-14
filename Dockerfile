# Use an official Ubuntu base image
FROM ubuntu:22.04

# prevent interactive prompts, (needed when installing python3.11, which installs tazdata which asks for geographic area)
ARG DEBIAN_FRONTEND=noninteractive

# update and install basic packages
RUN apt-get update && \
    apt-get install -y software-properties-common wget nano bzip2 build-essential tar


# install Python 3.11 stuff
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3.11-venv python3.11-distutils python3-pip python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# set Python 3.11 as the default version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 &&\
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 &&\
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# set the working directory inside the container
WORKDIR /scalable-multimodal-similarity-search

# copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt



# install CMake 3.29.2 through official binary
ARG CMAKE_VERSION=3.29.2
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzvf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    mv cmake-${CMAKE_VERSION}-linux-x86_64 /opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/bin/cmake && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz


#install pybind11 through conda
# Install Miniconda (Minimal Conda distribution)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && /opt/conda/bin/conda clean -all -y

# Make conda available globally
ENV PATH="$PATH:/opt/conda/bin:"

# Update conda and add conda-forge channel
RUN conda update -n base --all -c conda-forge -c defaults -y && \
    conda install -c conda-forge pybind11


# copy the rest of the application code to the working directory
COPY . .

CMD ["bash"]
