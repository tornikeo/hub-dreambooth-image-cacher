FROM huggingface/transformers-pytorch-cpu:latest
RUN apt update && apt-get install git-lfs && git lfs install
WORKDIR /workdir
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY main.py .
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}
CMD python3 main.py

