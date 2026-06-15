FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py .

EXPOSE 8765

ENV PYTHONUNBUFFERED=1

CMD ["python", "proxy.py"]
