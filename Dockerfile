FROM satyajitghana/pytorch:2.3.1

# Set environment variables
ENV UVICORN_SERVER_PORT=8080

# Set the working directory inside the container
WORKDIR /var/task
# Copy requirements file and install dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the application code into the container
COPY app.py ./
COPY model.pt ./
COPY examples/ ./examples/
COPY templates/ ./templates/
# Expose the FastAPI port
EXPOSE 8080

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]