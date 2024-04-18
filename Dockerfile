FROM --platform=linux/amd64 python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./glove.6B.50d.txt /code/glove.6B.50d.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app /code/app

EXPOSE 80
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80" ]
