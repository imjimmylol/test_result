FROM  python:3.10.0

ADD . /app

WORKDIR /app

RUN python -m pip install -r requirements.txt

CMD ["/bin/bash"]
