FROM python:3.9-slim-buster

WORKDIR /home/user/app

RUN apt-get install tesseract-ocr

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["streamlit", "run", "app.py"]