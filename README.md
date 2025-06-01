# VisionChat AI

**VisionChat AI** is an advanced AI-powered chatbot that performs **text analysis** using the **Mistral 7B** model and **image feature extraction** using **MMProg**. The extracted features are converted into numerical data, which is processed by **Mistral 7B** to generate meaningful responses based on user queries.

## Features

- **Text Processing:** Utilizes the **Mistral 7B** model for natural language understanding and response generation.
- **Image Analysis:** Uses **MMProg** for feature extraction from images.
- **Multi-Modal Projection:** Converts extracted features into a format that Mistral 7B can interpret for context-aware responses.
- **Chat History:** Maintains user interactions for improved conversation flow.
- **Streamlit Interface:** Provides an easy-to-use frontend for interaction.

## Installation & Setup

### **1. Clone the Repository**

```bash
https://github.com/Balaji1472/VisionChat-AI.git
cd VisionChat-AI
```

### **2. Set Up a Virtual Environment (Optional but Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate    # For Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Download Required Models**

Ensure you have the necessary models downloaded in your working directory:

- **Mistral 7B** ([Download Here](https://huggingface.co/mistralai/Mistral-7B-v0.1))
- **LLaMA** ([Download Here](https://huggingface.co/meta-llama/Llama-2-7b-hf))
- **MMProg** ([Download Here](https://huggingface.co/path/to/mmprog))

Place the models in the appropriate directories (e.g., `models/mistral`, `models/llama`, `models/mmprog`).

### **5. Run the Application**

```bash
streamlit run app1.py
```

## How It Works

1. **User Inputs an Image & Query:**
   - The chatbot accepts an image and a related textual query.
2. **Image Processing via MMProg:**
   - MMProg extracts numerical features from the image.
3. **Feature Transfer to Mistral 7B:**
   - The extracted features are passed to Mistral 7B for context-aware text analysis.
4. **Response Generation:**
   - The chatbot analyzes the prompt and provides an intelligent response based on both the image and user query.

## Contributing

Feel free to fork the repo, submit issues, and create pull requests. Any contributions to enhance Image Insight AI are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For queries and collaborations, reach out to **Balaji Vasu** via [GitHub](https://github.com/Balaji1472).

