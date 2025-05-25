import os
import gc
import psutil
import threading
import time
from llama_cpp import Llama
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS, cross_origin
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://lifelink-theta-cyan.vercel.app", "https://lifelink-app.vercel.app/"])

model_path = os.getenv("MODEL_PATH", "./Mistral-7B-Instruct-v0.3.IQ1_S.gguf")

# Global model instance with lazy loading and memory management
model = None
model_lock = threading.Lock()
last_used = 0
MODEL_TIMEOUT = 180  # Unload model after 3 minutes of inactivity (reduced)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    gc.collect()
    gc.collect()
    # Force garbage collection of unreachable objects
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

def get_model():
    global model, last_used

    with model_lock:
        current_time = time.time()

        # Check if model exists and hasn't timed out
        if model is not None:
            last_used = current_time
            return model

        # Check available memory before loading
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        print(f"Available memory: {available_memory:.1f}MB")

        if available_memory < 500:
            force_cleanup()
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            if available_memory < 500:
                raise Exception(f"Insufficient memory: {available_memory:.1f}MB available, need 600MB+")

        print(f"Loading model... Memory before: {get_memory_usage():.1f}MB")

        try:
            # Ultra-minimal model configuration for t4g.small
            model = Llama(
                model_path=model_path,
                n_ctx=256,           # Drastically reduced context window
                n_threads=2,        # Single thread
                n_batch=8,          # Process one token at a time
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                verbose=False,
                cache=True,
                use_mmap=True,      # Use memory mapping
                use_mlock=False,    # Don't lock pages in memory
                f16_kv=True,        # Use 16-bit for key-value cache
                n_gpu_layers=0,     # No GPU offloading
                offload_kv_cache=False,
                rope_scaling_type=-1,
                rope_freq_base=0,
                rope_freq_scale=0,
                low_vram=True,
                # Additional memory-saving options
                numa=False,
                embedding=False,
                last_n_tokens_size=32,
            )

            last_used = current_time
            print(f"Model loaded successfully. Memory after: {get_memory_usage():.1f}MB")
            return model

        except Exception as e:
            model = None
            force_cleanup()
            print(f"Failed to load model: {e}")
            raise Exception(f"Model loading failed: {str(e)}")

def unload_model():
    """Unload model to free memory"""
    global model
    with model_lock:
        if model is not None:
            print(f"Unloading model... Memory before: {get_memory_usage():.1f}MB")
            try:
                # Try to explicitly close model resources
                if hasattr(model, 'close'):
                    model.close()
                if hasattr(model, '_model') and hasattr(model._model, 'close'):
                    model._model.close()
            except:
                pass
            del model
            model = None
            force_cleanup()
            print(f"Model unloaded. Memory after: {get_memory_usage():.1f}MB")

def memory_monitor():
    """Background thread to monitor memory and unload model if needed"""
    while True:
        try:
            time.sleep(20)  # Check every 20 seconds

            # Check if model should be unloaded due to inactivity
            if model is not None and (time.time() - last_used) > MODEL_TIMEOUT:
                print("Unloading model due to inactivity")
                unload_model()
                continue

            # Check memory pressure - more aggressive
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:  # If using more than 80% of memory
                print(f"High memory usage: {memory_percent:.1f}%. Forcing cleanup...")
                if model is not None:
                    unload_model()
                else:
                    force_cleanup()

        except Exception as e:
            print(f"Memory monitor error: {e}")

# Start memory monitor thread
monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
monitor_thread.start()

def calculate_age(dob: str) -> int:
    try:
        birthdate = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except:
        return 30

def build_user_context(profile: dict) -> str:
    age = calculate_age(profile.get('dob', '1990-01-01'))
    gender = profile.get("gender", "person").lower()
    firstName = profile.get("first_name", "User")
    conditions = profile.get("medical_conditions", "").strip()
    conditions = conditions if conditions and conditions.lower() != "none" else "none"
    medications = profile.get("medications", "").strip()
    medications = medications if medications and medications.lower() != "none" else "none"

    # Shortened context to save tokens
    return f"{age}y {gender[0].upper()}: {conditions[:20]}, {medications[:20]}"

def build_prompt(user_context: str, user_question: str) -> str:
    # Very short prompt to maximize response space
    return f"[INST]{user_context}\nQ:{user_question}\nGive brief medical advice:[/INST]"

@app.route('/health', methods=['GET'])
def health_check():
    memory_percent = psutil.virtual_memory().percent
    memory_mb = get_memory_usage()
    model_loaded = model is not None

    return jsonify({
        'status': 'healthy',
        'memory_usage_percent': f"{memory_percent:.1f}%",
        'memory_usage_mb': f"{memory_mb:.1f}MB",
        'model_loaded': model_loaded
    }), 200

@app.route('/chat/stream', methods=['POST'])
@cross_origin()
def stream_chat():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON in request'}), 400

        prompt_text = data.get('newPrompt', '')
        user_profile = data.get('userProfile', {})

        if not prompt_text.strip():
            return jsonify({"error": "Empty prompt"}), 400

        # Build context and prompt
        user_context = build_user_context(user_profile)
        prompt = build_prompt(user_context, prompt_text)

        print(f"Processing request. Memory usage: {get_memory_usage():.1f}MB")
        print(f"Prompt length: {len(prompt)} chars")

        # Get model instance
        try:
            model_instance = get_model()
        except Exception as e:
            return jsonify({"error": f"Model unavailable: {str(e)}"}), 503

        def generate():
            try:
                token_count = 0
                max_tokens = 150  # Very short responses to fit in context
                response_text = ""

                for chunk in model_instance(
                    prompt=prompt,
                    stream=True,
                    max_tokens=max_tokens,
                    stop=["[INST]", "</s>", "[/INST]", "\n\n", "Q:", "Question:"],
                    echo=False,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                ):
                    if token_count >= max_tokens:
                        break

                    token = chunk["choices"][0]["text"]
                    if token and token.strip():
                        response_text += token
                        # Send as server-sent events format for proper streaming
                        yield f"data: {token}\n\n"
                        token_count += 1

                        # Yield control occasionally
                        if token_count % 10 == 0:
                            time.sleep(0.005)

                # Send end marker
                yield f"data: [DONE]\n\n"

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {error_msg}\n\n"
                yield f"data: [DONE]\n\n"
            finally:
                # Force cleanup after each request to prevent memory buildup
                force_cleanup()

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
            }
        )

    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

@app.teardown_appcontext
def cleanup(exception):
    force_cleanup()

# Graceful shutdown
import signal
import sys

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    unload_model()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print(f"Starting with {psutil.virtual_memory().total / 1024 / 1024:.0f}MB total memory")
    app.run(debug=False, port=8080, host='0.0.0.0', threaded=True)