FROM python:3.8.13-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg 

RUN pip install --upgrade setuptools 

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD . .

EXPOSE 8000:8000

CMD uvicorn app:app --host 0.0.0.0 --port 8000