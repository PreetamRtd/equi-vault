# Use Python 3.11 (Highly stable for Data Science Linux wheels)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Change working directory to backend so python finds local imports
WORKDIR /app/backend

# Run the FastAPI server on port 8080 (Required by Google Cloud Run & Render)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]