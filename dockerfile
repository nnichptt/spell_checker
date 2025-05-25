FROM python:3.11-slim

# Set working directory
WORKDIR .

# Copy your app code into the container
COPY . .

# Install any dependencies (adjust if needed)
RUN pip install -r requirements.txt

# Expose internal container port (common for Flask: 5000)
EXPOSE 5000

# Command to run your app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
