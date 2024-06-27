# Use the official Python image as your base
FROM python:3.11.0

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install dependencies from the requirements file
RUN pip install -r requirements.txt

# Copy your application code
COPY . /app

# Expose the port your application will run on (e.g., 8501 for Streamlit)
EXPOSE 8501

# Command to run your application (replace with your actual command)
CMD ["streamlit", "run", "main.py"]
