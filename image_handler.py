from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    try:
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return "data:image/jpeg;base64," + encoded_string
    except Exception as e:
        print(f"Error in convert_bytes_to_base64: {str(e)}")
        raise

def handle_image(image_bytes, user_message):
    try:
        chat_handler = Llava15ChatHandler(clip_model_path="./models/llava/mmproj-model-f16.gguf")
        llm = Llama(
            model_path="./models/llava/ggml-model-q5_k.gguf",
            chat_handler=chat_handler,
            logits_all=True,
            n_ctx=2048  # Increased context window
        )
        
        image_base64 = convert_bytes_to_base64(image_bytes)

        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": user_message}
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=512
        )
        
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in handle_image: {str(e)}")
        return f"Error processing image: {str(e)}"

def convert_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return "data:image/jpeg;base64," + encoded_string
    except Exception as e:
        print(f"Error in convert_image_to_base64: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = "Image26.jpg"
    image_base64 = convert_image_to_base64(image_path)
    with open("image.txt", "w") as f:
        f.write(image_base64)