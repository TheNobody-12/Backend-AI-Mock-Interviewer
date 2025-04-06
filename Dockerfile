FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy and export environment variables from .env using dotenv
RUN pip install python-dotenv

# Expose Flask port
EXPOSE 8080

CMD ["python", "app.py"]
