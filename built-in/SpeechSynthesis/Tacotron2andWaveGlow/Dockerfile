ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.6.0-torch1.6-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

RUN mkdir -p /home/pytorch_modelzoo/

COPY ./tools /home/pytorch_modelzoo/built-in/tools

COPY ./built-in/speechsynthesis/ /home/pytorch_modelzoo/built-in/SpeechSynthesis
WORKDIR /home/pytorch_modelzoo/built-in/SpeechSynthesis

ENV VIRTUAL_ENV=/torch/venv3/pytorch
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN pip install -r models/requirements.txt

