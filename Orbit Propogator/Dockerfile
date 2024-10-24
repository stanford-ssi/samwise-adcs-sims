FROM public.ecr.aws/lambda/python:3.9
WORKDIR ${LAMBDA_TASK_ROOT}

# Install the required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the contents of src and lambda_function.py into the container
COPY src/ ./src/
COPY main.py .
COPY lambda_function.py .

# Specify the handler for AWS Lambda
CMD ["lambda_function.propagate"]