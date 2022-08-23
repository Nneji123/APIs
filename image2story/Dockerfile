FROM python:3.8.13-slim-bullseye

WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
  git

RUN pip install --upgrade setuptools 

COPY . .

RUN pip install -r requirements.txt

RUN python download.py

EXPOSE 8000:8000

CMD uvicorn app:app --reload --host 0.0.0.0 --port 8000