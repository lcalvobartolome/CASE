FROM python:3.9

WORKDIR /case-inferencer

COPY . .

RUN mkdir -p /data/source
RUN mkdir -p /data/inference

RUN python3 -m pip install -r requirements.txt

RUN apt-get update

# Define timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install jre
RUN apt install -y default-jre

# Install wget and unzip
RUN apt-get update && apt-get install -y wget unzip

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
RUN export JAVA_HOME

# Download Mallet
RUN wget https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip

# Extract the contents of the zip file
RUN unzip mallet-2.0.8.zip

# Remove the zip file
RUN rm mallet-2.0.8.zip

# Remove old mallet if exists
RUN rm -rf /case-inferencer/src/core/models/mallet-2.0.8

# Copy the Mallet's decompressed folder into the Mallet directory
RUN mv mallet-2.0.8 /case-inferencer/src/core/models

EXPOSE 90

CMD ["python3", "app.py"]