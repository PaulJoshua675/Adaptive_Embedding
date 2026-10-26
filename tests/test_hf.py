import os
from huggingface_hub import InferenceClient

def test_hf_connection():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN environment variable is not set.")
        return False

    print("🔑 HF_TOKEN found. Initializing InferenceClient...")

    try:
        # Initialize the official client
        client = InferenceClient(api_key=token)

        # Test text generation using a lightweight model to verify connection & auth
        response = client.text_generation(
            model="gpt2",
            prompt="Hello, connection test:",
            max_new_tokens=10
        )
        
        print("✅ Success! Connected to Hugging Face successfully.")
        print(f"📥 Response received: {response.strip()}")
        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_hf_connection()