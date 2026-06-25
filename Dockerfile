FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 7860

# Start application with uvicorn
CMD ["uvicorn", "app_spaces:app", "--host", "0.0.0.0", "--port", "7860"]
