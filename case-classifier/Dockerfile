FROM python:3.9

WORKDIR /case-classifier

COPY . .

RUN mkdir -p /data/classifier

RUN python3 -m pip install torch
RUN python3 -m pip install -r requirements.txt 

EXPOSE 100

CMD ["python3", "app.py"]