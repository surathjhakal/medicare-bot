import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.chatbot import ChatBot
from utils.upload_file import UploadFile

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    chatbot=request.form['chat-history'] # [[question,answer],[question,answer]]
    message = request.form['question']
    data_type=request.form['data_type']
    temperature=0.0

    response,references=ChatBot.respond(chatbot,message,data_type,temperature)
    return jsonify({'response': response,'references':references})

@app.route('/upload-doc', methods=['POST'])
def uploadPDF():
    file = request.files['file']
    save_path="data/docs_2/"+file.filename
    print(save_path)
    file.save(save_path)

    rag_with_dropdown="Upload doc: Process for RAG"
    local_file_path = "C:/Users/jhaka/Desktop/medical-rag-chatbot/backend/data/docs_2/"+file.filename
    print(local_file_path)
    response=UploadFile.process_uploaded_files([local_file_path],rag_with_dropdown)

    return jsonify({'response': response})

@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files['file']
    save_path="data/docs_2/"+file.filename
    print(save_path)
    file.save(save_path)

    rag_with_dropdown="Upload doc: Give full Summary"
    local_file_path = "C:/Users/jhaka/Desktop/medical-rag-chatbot/backend/data/docs_2/"+file.filename

    response=UploadFile.process_uploaded_files([local_file_path],rag_with_dropdown)

    return jsonify({'response': response})

if __name__ == '__main__':
  app.run(debug=True)

