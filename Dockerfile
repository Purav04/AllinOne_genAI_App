FROM docker.io/library/python:3.10@sha256:eb7df628043d68aa30019fed02052bd27f1431c3a0abe9299d1e4d804d4b11e0

RUN apt-get install tesseract-ocr

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

CMD ["streamlit", "run", "app.py"]