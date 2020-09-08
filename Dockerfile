FROM python:3.8.5
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 2000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]