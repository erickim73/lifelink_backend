from llama_cpp import Llama
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
from datetime import datetime


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

model_path = "./Mistral-7B-Instruct-v0.3.Q5_K_S.gguf"

model = Llama(model_path=model_path, n_ctx=32768, n_threads=12, n_batch=64, temperature=0.7, top_p=0.95, repeat_penalty=1.2, verbose=False, cache=True)

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
        "[INST]\n"
        # ─── Role ────────────────────────────────────────────────────────────
        "You are a helpful, empathetic AI Medical Assistant. Address the user "
        "as “you,” refer to yourself as “I,” and avoid medical jargon.\n\n"

        # ─── Formatting rules ───────────────────────────────────────────────
        "FORMAT RULES\n"
        "1. When you give a numbered list, write it like:\n"
        "   1. First item text\n\n"
        "   2. Second item text\n\n"
        "   3. Third item text\n\n"
        "   (← exactly two \\n after every item, including the last.)\n"
        "2. Do **not** insert single newlines inside an item.\n"
        "3. After the list, continue with a normal paragraph.\n\n"

        # ─── Emergency rule ─────────────────────────────────────────────────
        "If you detect possible emergencies (heart attack, stroke, chest pain, "
        "trouble breathing, excessive bleeding), STOP and reply only with:\n"
        "\"This may be a medical emergency. Please call 911 or go to the ER immediately.\"\n\n"

        # ─── User context & question ────────────────────────────────────────
        f"User context: {user_context}\n"
        f"Question: {user_question}\n"
        "[/INST]"
    )

    

    
@app.route('/chat/stream', methods=['POST'])
@cross_origin()
def stream_chat():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'Invalid JSON in request'}), 400
    
    prompt_text = data.get('newPrompt', '')
    user_profile = data.get('userProfile')
    
    if not user_profile or not isinstance(user_profile, dict):
        return jsonify({"error": "Invalid or missing user profile"}), 400
    
    try:
        user_context = build_user_context(user_profile)
    except Exception as e:
        return jsonify({"error": f"Failed to build user context: {str(e)}"}), 400
    
    
    prompt = build_prompt(user_context, prompt_text)
    
    flush_triggers = {".", "!", "?"}
    MAX_CHARS      = 400
    
    def generate():
        for chunk in model(prompt=prompt, stream=True, max_tokens=32768):
            text_chunk = chunk["choices"][0]["text"]
            yield f"data: {text_chunk}\n\n"
        yield "data: [DONE]\n\n"
        
    return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
if __name__ == '__main__':
    app.run(debug=True, port=8080)