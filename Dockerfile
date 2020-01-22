FROM python:3.6.9
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python server.py
