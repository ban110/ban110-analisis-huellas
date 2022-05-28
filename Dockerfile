FROM ubuntu:18.04

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

RUN apt purge  -y python-pip python3-pip
RUN rm -rf /usr/local/lib/python2.7/dist-packages/pip
RUN rm -rf /usr/local/lib/python3.4/dist-packages/pip
RUN apt install -y python-pip python3-pip

RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
# We copy just the  requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install opencv-python

RUN pip3 install -r requirements.txt

COPY . /app

WORKDIR /app



EXPOSE 5000

RUN mkdir received

ENTRYPOINT [ "python3" ]

CMD [ "main.py" ]
