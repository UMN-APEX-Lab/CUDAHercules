# Evaluation image of class3/long_context_ttft: vLLM 0.30.0 (CUDA 13, PyTorch 2.13) plus the
# packages the reference implementation needs. Model weights are mounted at runtime:
#   docker build -t hercules-lct -f docker/Dockerfile .
#   docker run --gpus '"device=0"' -v $HF_HOME:/hf -e HF_HOME=/hf -v <data>:/data -e HERCULES_DATA_DIR=/data \
#       -v $PWD:/task -w /task hercules-lct python run.py candidate --plugin src
FROM vllm/vllm-openai:v0.30.0
COPY reference/requirements.txt /tmp/reference-requirements.txt
RUN pip install --no-cache-dir -r /tmp/reference-requirements.txt
ENV HERCULES_VLLM_PYTHON=python3
ENTRYPOINT []
