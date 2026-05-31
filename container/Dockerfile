FROM python:3.11-slim

WORKDIR /app


COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY container/app/ .


COPY container/models/ ./models/


COPY container/STUDENT.json .


ENTRYPOINT ["python", "cli.py"]