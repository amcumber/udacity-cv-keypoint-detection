# # Modified from https://github.com/salvioli/deep-rl-tennis
FROM ubuntu:latest
LABEL author="aaron.mcumber@gmail.com"
LABEL citation="https://u.group/thinking/how-to-put-jupyter-notebooks-in-a-dockerfile/"

RUN apt-get update && apt-get -y update
RUN apt-get install --no-install-recommends -y build-essential python3.9 && \
    python3-pip python3-dev
RUN apt-get install -y unzip git
RUN apt-get install wget

RUN pip3 install -q pip --upgrade && \
    pip3 install jupyter ipykernel numpy matplotlib torch torchvision torchaudio

RUN mkdir /workspace
RUN mkdir -p /data && \
    cd /workspace

WORKDIR /workspace/

COPY . .

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root"]
