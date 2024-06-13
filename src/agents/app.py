import json
import logging
import time
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
import os
import base64
from uagents.query import query
from uagents import Agent, Context, Model, Bureau
from uagents.setup import fund_agent_if_low
 
app = Flask(__name__)
load_dotenv()
 
# Configuration
GPT4V_KEY = os.getenv('GPT4V_KEY')
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}
GPT4V_ENDPOINT = "https://cude-testing.openai.azure.com/openai/deployments/gpt4-with-vision/chat/completions?api-version=2024-02-15-preview"
 
# Define the input data model
class InputData(Model):
    content: str
 
# Define the result data model
class ResultData(Model):
    res: str
 
# Define the Message data model
 
class Message(Model):
    message: str
    address: str
 
 
summerizing_agent = Agent(name='summerizing_agent', seed="summerizing_agent recovery phrase",port=5001,endpoint=['http://localhost:5001/submit'])
extracting_agent = Agent(name='extracting_agent', seed="extracting_agent recovery phrase",port=5002,endpoint=['http://localhost:5002/submit'])
mapping_agent = Agent(name='mapping_agent', seed="mapping_agent recovery phrase",port=5003,endpoint=['http://localhost:5003/submit'])
recommending_agent = Agent(name='recommending_agent', seed="recommending_agent recovery phrase",port=5004,endpoint=['http://localhost:5004/submit'])
 
fund_agent_if_low(summerizing_agent.wallet.address())
fund_agent_if_low(extracting_agent.wallet.address())
fund_agent_if_low(mapping_agent.wallet.address())
fund_agent_if_low(recommending_agent.wallet.address())
 
summerizing_agent_address=summerizing_agent.address
 
#Summerization Function
def llm_call_summerizing(encoded_files, user_id, request_id):
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
            "max_tokens": 2000
        }
 
        result = make_request(GPT4V_ENDPOINT, payload)
        return result
    except Exception as e:
        return {"error": f"Error in processing: {str(e)}"}
 
 #
 
#Extracting Data
def llm_call_extracting(summary_text, user_id, request_id):
    try:
        content = summary_text
 
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
                            "text": """Extract the information in the ITEM_LIST from the SUMMARIZED_TEXT:
                                       ITEM_LIST:
                                       GENERAL INFO:
                                       - Full Scheme Name
                                       - Scheme Established
                                       - Contracted Out
                                       - Contracted Out Ended
                                       - Post 5 April 1997 Basis
                                       - Equalisation Date Scheme 1
                                       - Equalisation Date Scheme 2
                                       
                                       SECTION 1 to SECTION 8:
                                       - Any relevant info that needs mapping
                                       
                                       SUMMARIZED_TEXT:
                                    """
                        }
                    ] + content
                }
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2000
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
    print(f"Request time: {end_time - start_time} seconds")
    return response.json()['choices'][0]['message']['content']
 
@summerizing_agent.on_query(model=InputData,replies=ResultData)
async def qurey_handler(ctx:Context,sender:str, content:InputData):
    try:
       
        if not content:
            return jsonify({"error": "No file URL provided"}), 400
 
        try:
            downloaded_file = requests.get(content, headers={"User-Agent": "Mozilla/5.0"})
            downloaded_file.raise_for_status()
        except requests.exceptions.RequestException as e:
             return jsonify({"error": f"Failed to download file: {str(e)}"}), 400
 
        file_bytes = downloaded_file.content
        file_base64 = base64.b64encode(file_bytes).decode('utf-8')
        file_extension = content.split('.')[-1]
        file_data = {"type": file_extension, "data": file_base64}
 
        result = llm_call_summerizing([file_data], "user123", "request456")
        await ctx.send(extracting_agent.address,Message(message= result, address=sender))
 
 
    except Exception as e:
        error_message = f"Error! {str(e)}"
        await ctx.logger.info(error_message)
 
@extracting_agent.on_message(model=Message)
async def user_message_handler(ctx:Context, sender:str, incomingMessage:Message ):
 
 
    try:  
        logging.info("Calling GPT-4V with the downloaded file")
        result = llm_call_extracting(incomingMessage.message, "user123", "request456")
        logging.info("Received response from GPT-4V")
        await ctx.send(mapping_agent.address, Message(message= jsonify(result), address=incomingMessage.address))
 
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        return jsonify({"error": f"Error in processing: {str(e)}"}), 400
 
 
@mapping_agent.on_message(model=Message)
async def spice_message_handler(ctx: Context, sender: str, incomingMessage: Message):
   
 
    #Code
 
    await ctx.send(recommending_agent.address, Message(message= "output result", address=incomingMessage.address))
 
@recommending_agent.on_message(model=Message)
async def spice_message_handler(ctx: Context, sender: str, incomingMessage: Message):
   
 
    #Code
 
    await ctx.send(incomingMessage.address,  ResultData(res= "output result"))
 
@app.route('/identify', methods=['POST'])
async def identify():
    data = request.json
    file_url = data.get('file_url')
    response =  await query(destination=summerizing_agent_address, message=InputData(content=file_url), timeout=15.0)
    responseData = json.loads(response.decode_payload())
    return jsonify( responseData)
 
 
 
 
bureau = Bureau()
bureau.add(summerizing_agent)
bureau.add(extracting_agent)
bureau.add(mapping_agent)
bureau.add(recommending_agent)
 
if __name__ == "__main__":
    bureau.run()
 
if __name__ == '__main__':
    app.run(debug=True, port=5100)