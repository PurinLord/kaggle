FROM python:3.7-slim

WORKDIR /app

add ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . /app/

