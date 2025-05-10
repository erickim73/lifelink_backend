from llama_cpp import Llama
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
from datetime import datetime

app = Flask(__name__)
CORS(app)

model_path = "./mistral-7b-q4km.gguf"

model = Llama(model_path=model_path, n_ctx=2048, n_threads=8, n_batch=8, temperature=0.7, top_p=0.95, repeat_penalty=1.2, verbose=False)

def calculate_age(dob: str) -> int:
    birthdate = datetime.strptime(dob, "%Y-%m-%d")
    today = datetime.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

def build_user_context(profile: dict) -> str: 
    age = calculate_age(profile['dob'])
    gender = profile.get("gender").lower()
    firstName = profile.get("first_name")
    lastName = profile.get("last_name")
    conditions = profile.get("medical_conditions", "").strip()
    conditions = conditions if conditions and conditions.lower() != "none" else "no medical conditions"
    medications = profile.get("medications", "").strip()
    medications = medications if medications and medications.lower() != "none" else "no current medications"
    goals = profile.get("health_goals", "").strip()
    goals = goals if goals and goals.lower() != "none" else "no specific health goals"
    
    return (
        f"The user is a {age}-year-old {gender} named {firstName}. "
        f"They have {conditions.lower()}, are taking {medications}, and their health goal is to {goals.lower()}."
    )
    
def build_prompt(user_context: str, user_question: str) -> str:
    return (
        "[INST] You are a helpful medical assistant. "
        f"{user_context} "
        "Answer the user's question clearly and naturally.\n\n "
        f"### Question: {user_question} [/INST] "
    )
    
@app.route('/chat', methods=['POST'])
def generate_reply():
    data = request.get_json()
    prompt_text = data.get('newPrompt', '')
    user_profile = data.get('userProfile', {})  

    user_context = build_user_context(user_profile)
    prompt = build_prompt(user_context, prompt_text)

    output = model(prompt=prompt, max_tokens=512)
    return jsonify({
        "response": output['choices'][0]['text'].strip()
    })

    
@app.route('/chat/stream', methods=['POST'])
@cross_origin()
def stream_chat():
    data = request.get_json()
    prompt_text = data.get('newPrompt', '')
    user_profile = data.get("userProfile", {})
    
    user_context = build_user_context(user_profile)
    prompt = build_prompt(user_context, prompt_text)
    
    def generate():
        for chunk in model(prompt=prompt, stream=True, max_tokens=512):
            text_chunk = chunk["choices"][0]["text"]
            yield f"data: {text_chunk}\n\n"
        yield "data: [DONE]\n\n"
        
    return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
if __name__ == '__main__':
    app.run(debug=True, port=8080)