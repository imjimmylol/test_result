FROM  python:3.8

ADD . /app

WORKDIR /app

RUN python -m pip install -r requirements.txt

CMD python3 main.py
