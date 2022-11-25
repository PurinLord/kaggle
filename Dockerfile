FROM python:3.7-slim

WORKDIR /app

add ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . /app/

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "2469"]
