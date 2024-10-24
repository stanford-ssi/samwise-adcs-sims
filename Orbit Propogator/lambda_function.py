import json
from main import propagate_orbit  # Import the core function

def propagate(event, context):
    body = json.loads(event["body"]) # Get request body from the Lambda event
    result, status_code = propagate_orbit(body)  # Call the core logic function

    # Format the response for API Gateway
    return {
        "statusCode": status_code,
        "body": json.dumps(result),
        "headers": {
            "Content-Type": "application/json"
        }
    }