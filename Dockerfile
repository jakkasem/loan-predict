FROM python:3.11-slim

# Prevent Python from writing .pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port
EXPOSE 8000

# Run the application using uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
