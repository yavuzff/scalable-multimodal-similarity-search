# Use an official Ubuntu base image
FROM ubuntu:22.04

# Prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install essential tools and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        gpg-agent \
        nano \
        bzip2 \
        build-essential \
        tar \
        git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        python3-pip \
        python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install CMake 3.29.2
ARG CMAKE_VERSION=3.29.2
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzvf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    mv cmake-${CMAKE_VERSION}-linux-x86_64 /opt/cmake && \
    ln -s /opt/cmake/bin/cmake /usr/bin/cmake && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Install Catch2 for C++ testing (via git clone)
RUN git clone https://github.com/catchorg/Catch2.git &&\
    cd Catch2 &&\
    cmake -B build -S . -DBUILD_TESTING=OFF &&\
    cmake --build build/ --target install &&\
    cd .. &&\
    rm -rf Catch2

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean --all -y

# Add Conda to the PATH
ENV PATH="${PATH}:/opt/conda/bin"

# Install pybind11 via conda
RUN conda update -n base --all -c conda-forge -c defaults -y && \
    conda install -c conda-forge pybind11 -y


# Set working directory
WORKDIR /scalable-multimodal-similarity-search

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Install Valgrind for memory leak detection - for development
# Also install zlib1g-dev, to be used by cnpy for reading npy files
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      valgrind \
      zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# install cnpy for reading npy files
RUN git clone https://github.com/rogersce/cnpy.git \
    && cd cnpy \
    && cmake .\
    && make \
    && make install \
    && cd .. \
    && rm -rf cnpy

# install eigen for vector operations (about 5 mins to install))
RUN git clone https://gitlab.com/libeigen/eigen.git \
    && mkdir eigen-build \
    && cd eigen-build \
    && cmake ../eigen\
    && make install \
    && cd .. \
    && rm -rf eigen-build\
    && rm -rf eigen

# Copy the rest of the application code
COPY . .

# Compile the Cpp code and setup the bindings
#RUN cd cpp && \
#    cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Release && \
#    cmake --build cmake-build-debug -j 6 --target multimodal_index && \
#    python3 pybinding/setup.py install

# Default to bash
CMD ["bash"]
