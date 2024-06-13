from flask import Flask, request, render_template, redirect, jsonify
import requests
import os
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    file_url = None
    filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_url, filename = upload_file(file)
    return render_template('index.html', file_url=file_url, filename=filename, gpt_response=None)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    file_url = data.get('file_url')
    filename = data.get('filename')
    gpt_response = call_secondary_app(file_url, filename)
    return jsonify(gpt_response)

def upload_file(file):
    url = "https://file.io"
    response = requests.post(url, files={'file': file})
    if response.status_code == 200:
        return response.json()['link'], file.filename
    else:
        response.raise_for_status()

def call_secondary_app(file_url, filename):
    try:
        response = requests.post('http://127.0.0.1:5100/identify', json={'file_url': file_url, 'filename': filename})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(debug=True)
