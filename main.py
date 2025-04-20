from llama_cpp import Llama
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_path = "./mistral-7b-q4_0.gguf"

model = Llama(model_path=model_path, n_ctx=2048, n_threads=8, n_batch=8, temperature=0.7, top_p=0.95, repeat_penalty=1.2, verbose=False)


@app.route('/chat', methods=['POST'])
def generate_reply():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    output = model(prompt=prompt, max_tokens=128)
    return jsonify({
        "response": output['choices'][0]['text']
    })
    
if __name__ == '__main__':
    app.run(debug=True)
