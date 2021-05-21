FROM python:3.8-slim-buster

RUN apt-get update \
 && apt-get install -y libenchant-dev libgomp1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv
COPY Pipfile* /tmp/
RUN cd /tmp && pipenv lock --keep-outdated --requirements > requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . /app
WORKDIR /app

ENV FLASK_APP main.py
CMD ["flask", "run", "--host=0.0.0.0"]