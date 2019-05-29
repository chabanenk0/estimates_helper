FROM ubuntu:latest
MAINTAINER Dmytro Chabanenko "chdn6026@gmail.com"
EXPOSE 5000:5000
RUN apt-get update -y
RUN apt-get install -y python3 python3-dev python3-pip nginx
#RUN apt install -y python3-matplotlib
RUN pip3 install uwsgi
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["python3", "app.py"]
