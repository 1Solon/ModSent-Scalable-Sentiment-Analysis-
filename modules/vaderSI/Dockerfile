FROM python:latest

WORKDIR /vaderSI

# Install requirements from file
COPY ./requirements.txt /vaderSI
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

# Set environment variable, then run flask
ENV FLASK_APP=vaderAPI.py
CMD ["flask", "run", "--host", "0.0.0.0"]

# To run this dockerfile, use the below command:
# docker run -d -p 5000:5000 vader_si