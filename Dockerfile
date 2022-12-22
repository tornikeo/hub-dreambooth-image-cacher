FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN apt update && apt install git git-lfs wget libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 -y
RUN git lfs install
RUN conda install xformers -c xformers/label/dev -y
WORKDIR /workdir
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY main.py .

ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

ARG HF_HOME=/huggingface
ENV HF_HOME=${HF_HOME}

CMD python3 main.py