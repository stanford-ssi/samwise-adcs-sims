from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
from main import propagate_orbit  # Import the core function

app = Flask(__name__)
CORS(app)

@app.route("/propagate", methods=["POST"])
def propagate():
    body = request.get_json()  # Get request JSON data
    result, status_code = propagate_orbit(body)  # Call the core logic function
    return jsonify(result), status_code  # Return the result in Flask response format


if __name__ == '__main__':  
   app.run()