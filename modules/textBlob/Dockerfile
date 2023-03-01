FROM python:latest

WORKDIR /NLTK

# Install requirements from file
COPY ./requirements.txt /nltkAPI
RUN pip install -r requirements.txt

COPY . .

EXPOSE 3200

# Set environment variable, then run flask
ENV FLASK_APP=nltkAPI.py
CMD ["flask", "run", "--host", "0.0.0.0"]

# To run this dockerfile, use the below command:
# docker run -d -p 3300:5000 nltk