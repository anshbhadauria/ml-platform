FROM python:3.9
# conda/miniconda3
# prepare user able to read file results of this image

RUN mkdir {{ cookiecutter.project_name }}
RUN cd {{ cookiecutter.project_name }}

ADD conf {{ cookiecutter.project_name }}/conf
ADD artifacts {{ cookiecutter.project_name }}/artifacts
ADD infra {{ cookiecutter.project_name }}/infra
ADD src {{ cookiecutter.project_name }}/src

COPY run.py {{ cookiecutter.project_name }}/
COPY requirements.txt {{ cookiecutter.project_name }}/

ARG HTTPS_PROXY

# Init PIP user env
ARG PIP_INDEX
ARG PIP_HOST
RUN echo "[global]" > /pip.conf
RUN echo "index-url = $PIP_INDEX" >> /pip.conf
RUN echo "trusted-host = $PIP_HOST" >> /pip.conf
ENV PIP_CONFIG_FILE=/pip.conf

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN mkdir ~/.aws
RUN echo "[{{ cookiecutter.aws_profile }}]" > ~/.aws/credentials
RUN echo "aws_access_key_id = $AWS_ACCESS_KEY_ID" >> ~/.aws/credentials
RUN echo "aws_secret_access_key = $AWS_SECRET_ACCESS_KEY" >> ~/.aws/credentials

RUN pip3 install -r {{ cookiecutter.project_name }}/requirements.txt
WORKDIR {{ cookiecutter.project_name }}

ENTRYPOINT [ "python", "./src/models/train.py", "--i", "--o"]
