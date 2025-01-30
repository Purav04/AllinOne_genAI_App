FROM python:3.9-slim-buster

RUN apt-get install tesseract-ocr

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["streamlit", "run", "app.py"]