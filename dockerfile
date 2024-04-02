FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt dist-upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:alex-p/tesseract-ocr5
RUN apt install -y tesseract-ocr
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
COPY . /APIANPR
WORKDIR /APIANPR
RUN pip3 install -r requirements.txt
RUN pip3 install flask
CMD [ "python3" ,"app.py" ]
EXPOSE 5001
