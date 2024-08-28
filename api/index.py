import os
from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

app = Flask(__name__)

# Load the tokenizer and model dynamically to avoid large deployment packages
model_name = 'meta-llama/LLaMA-7B'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

@app.route('/send', methods=['POST'])
def send():
    user_input = request.json.get('message')
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
