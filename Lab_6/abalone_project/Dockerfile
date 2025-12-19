# Use an official Python runtime as a parent image
FROM python:3.12-slim


# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY app.py eda.py model.py requirements.txt ./

COPY data/abalone.data data/abalone.data

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container (for web apps - 80 or 8080)
EXPOSE 5000

RUN python eda.py

RUN python model.py

# Define environment variable (optional)
# ENV NAME World

# Run app.py when the container launches
CMD ["python", "./app.py"]
