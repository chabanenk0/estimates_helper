FROM ubuntu:latest
MAINTAINER Dmytro Chabanenko "chdn6026@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-pip nginx
RUN pip3 install uwsgi
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["python", "/app/app.py"]
