import time
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import base64

app = Flask(__name__)
load_dotenv()

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
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        response = requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400

    image_bytes = response.content
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data = {"type": "jpg", "data": image_base64}  # Assuming JPG, adjust as necessary

    result = llm_call_identify_items([image_data], "user123", "request456")
    return jsonify(result)

def llm_call_identify_items(encoded_images, user_id, request_id):
    try:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{encoded_image['type']};base64,{encoded_image['data']}"
                }
            } for encoded_image in encoded_images
        ]

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """            
                    You are an AI system with the experience of a recycling materials operator, able to make the best categorization, sorting, and selection of recycling items. 
                    Your primary expertise is in correctly identifying recycling items from the provided images.
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            # "text": "Identify recycling items in the attached images. Return it as a bulletpoint list only. The bulletpoint character should be the hash '#'. Do not put image names or any other elements which are not recyclable items. The items will be of 2 types: paper/cardboard and tin/can/plastics. If no recyclable items detected, return # N/A"
                            "text": "Identify recycling items in the attached images. Only recognize cans, tins, plastic, paper and cardboard. You have 3 return options and nothing else, based on the identified item(s): #Can&Plastic #Cardboard&Paper #UnIdentified"
                        }
                    ] + content
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2000
        }

        result = make_request(GPT4V_ENDPOINT, payload)
        return result
    except Exception as e:
        return {"error": f"Error in processing: {str(e)}"}

def make_request(GPT_ENDPOINT, payload):
    start_time = time.time()  # Start timing before the request
    response = requests.post(GPT_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    end_time = time.time()  # End timing after the response
    print(f"Request time: {end_time - start_time} seconds")
    return response.json()['choices'][0]['message']['content']

if __name__ == '__main__':
    app.run(debug=True)
