# Use an official Ubuntu base image
FROM ubuntu:22.04

# Install the necessary python packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /scalable-multimodal-similarity-search

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Create a virtual environment and install the required Python packages
RUN python3 -m venv /venv
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Set the environment variables
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/scalable-multimodal-similarity-search"

CMD ["bash"]
