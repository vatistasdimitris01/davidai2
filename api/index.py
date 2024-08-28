from flask import Flask, request, jsonify, send_from_directory
import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

# Load the LLaMA model and tokenizer
model_name = 'meta-llama/LLaMA-7B'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Serve the HTML page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Serve the CSS file
@app.route('/styles.css')
def styles():
    return send_from_directory('static', 'styles.css')

# Endpoint for sending messages
@app.route('/send', methods=['POST'])
def send():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'reply': 'Please provide a message.'}), 400

    # Tokenize and generate response
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'reply': reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
