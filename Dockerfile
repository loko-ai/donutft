FROM python:3.10-slim
ARG user
ARG password
ADD requirements.lock /
RUN pip install --upgrade --extra-index-url https://$user:$password@distribution.livetech.site -r /requirements.lock
ADD . /donutft
ENV PYTHONPATH=$PYTHONPATH:/donutft
WORKDIR /donutft/donutft/services
CMD python services.py
