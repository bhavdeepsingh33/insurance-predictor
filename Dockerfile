FROM python:3.6.10
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
#EXPOSE 5000
#Start Flask Server
#CMD [ "python","app.py"]
#Expose server port
CMD gunicorn app:app --bind 0.0.0.0:$PORT --reload

