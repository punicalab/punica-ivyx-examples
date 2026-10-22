# The image a Compute target runs train_ner.py in, built once on the
# machine that trains:
#
#   docker build -t ivyx-ner compute/
#
# The default base carries torch with CUDA and needs an NVIDIA GPU and the
# NVIDIA container toolkit on that machine. For a machine with no GPU:
#
#   docker build -t ivyx-ner --build-arg BASE=python:3.12-slim compute/
#
# Only training runs on the other machine. The ONNX export and both gates run
# here, so onnx and onnxruntime are not in the image.
ARG BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE}

# A slim base has no torch; a pytorch base keeps its own, which matches its CUDA.
RUN python -c "import torch" 2>/dev/null \
 || pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# transformers needs torch 2.6 or newer, which the default base has.
RUN pip install --no-cache-dir transformers numpy

# The base model goes into the image, so a job downloads nothing: every job
# starts in a fresh container, and a download would be repeated for each one.
# Keep it the same as BASE_MODEL in src/train_ner.py.
ARG BASE_MODEL=distilbert-base-multilingual-cased
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import AutoModelForTokenClassification as M, AutoTokenizer as T; T.from_pretrained('${BASE_MODEL}'); M.from_pretrained('${BASE_MODEL}')"
ENV HF_HUB_OFFLINE=1
