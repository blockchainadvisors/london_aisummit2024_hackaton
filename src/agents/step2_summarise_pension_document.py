import time
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import base64
import logging

app = Flask(__name__)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
GPT4V_KEY = os.getenv('GPT4V_KEY')
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}
GPT4V_ENDPOINT = "https://cude-testing.openai.azure.com/openai/deployments/gpt4-with-vision/chat/completions?api-version=2024-02-15-preview"

@app.route('/identify', methods=['POST'])
def identify():
    data = request.json
    file_url = data.get('file_url')
    filename = data.get('filename')

    if not file_url:
        logging.error("No file URL provided")
        return jsonify({"error": "No file URL provided"}), 400

    try:
        logging.info(f"Attempting to download file from URL: {file_url}")
        response = requests.get(file_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        logging.info("File downloaded successfully")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download file: {str(e)}")
        return jsonify({"error": f"Failed to download file: {str(e)}"}), 400

    try:
        file_bytes = response.content
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')
        logging.info("filename",filename)
        file_extension = filename.split('.')[-1]
        logging.info("file_extension",file_extension)
        file_data = {"type": file_extension, "data": file_base64}
        
        logging.info("Calling GPT-4V with the downloaded file")
        result = llm_call_identify_items([file_data], "user123", "request456")
        logging.info("Received response from GPT-4V")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return jsonify({"error": f"Error in processing: {str(e)}"}), 400

def llm_call_identify_items(encoded_files, user_id, request_id):
    try:
        content = [
            {
                "type": "file",
                "file": {
                    "url": f"data:application/{encoded_file['type']};base64,{encoded_file['data']}"
                }
            } for encoded_file in encoded_files
        ]

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """            
                    You are a helpful agent who can extract relevant information such as scheme details, retirement dates, pension benefits, etc., from a PDF written by an insurance agency about different pension schemes for other companies' inquiries.
 
                    Please follow these guidelines:
                        1. Try to answer the question as accurately as possible, using only reliable sources.
                        2. Rate your confidence in the accuracy of your answer from 0 to 1 based on the credibility of the data publisher and how much it might have changed since the publishing date.
                        3. In the last line of your response, provide the information in the exact JSON format: {"value": value, "unit": unit, "timestamp": time, "confidence": rating, "source": ref, "notes": summary}
                            - value is the numerical value of the data without any commas or units
                            - unit is the measurement unit of the data if applicable, or an empty string if not applicable
                            - time is the approximate timestamp when this value was published in ISO 8601 format
                            - rating is your confidence rating of the data from 0 to 1
                            - ref is a url where the data can be found, or a citation if no url is available
                            - summary is a brief justification for the confidence rating (why you are confident or not confident in the accuracy of the value)
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summarize the attached pension scheme documents."
                        }
                    ] + content
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 10000
        }

        logging.debug(f"Sending payload to GPT-4V: {payload}")
        result = make_request(GPT4V_ENDPOINT, payload)
        logging.debug(f"Received response from GPT-4V: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in LLM call: {str(e)}")
        return {"error": f"Error in processing: {str(e)}"}

def make_request(GPT_ENDPOINT, payload):
    start_time = time.time()  # Start timing before the request
    response = requests.post(GPT_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    end_time = time.time()  # End timing after the response
    logging.info(f"Request time: {end_time - start_time} seconds")
    return response.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    app.run(debug=True, port=5100)
