# MedBud: End-to-End Medical Chatbot Using Llama 2

MedBud is an advanced medical chatbot designed to assist users with medical inquiries. Built using the Llama 2 language model, FAISS for vector search, and Flask for the web interface, MedBud provides accurate and context-aware responses to user queries.

---

## 🚀 Features

- **Llama 2 Integration**: Utilizes the Llama 2 model for natural language understanding and response generation.
- **FAISS Vector Search**: Efficiently retrieves relevant information from indexed medical documents.
- **PDF Document Support**: Automatically processes and indexes medical PDFs for knowledge retrieval.
- **Web Interface**: User-friendly interface built with Flask for seamless interaction.
- **Customizable Prompts**: Easily modify prompts to fine-tune chatbot behavior.

---

## 🛠️ Technologies Used

- **[Llama 2](https://huggingface.co/meta-llama)**: Open-source large language model.
- **[FAISS](https://github.com/facebookresearch/faiss)**: Fast vector search library for document retrieval.
- **[Flask](https://flask.palletsprojects.com/)**: Lightweight web framework for the chatbot interface.
- **[LangChain](https://langchain.com/)**: Framework for building LLM-powered applications.
- **Python**: Core programming language.

---

## 📂 Project Structure

```
MedBud/
├── src/
│   ├── helper.py          # Helper functions for PDF loading, text splitting, and embeddings
│   ├── prompt.py          # Custom prompts for the chatbot
├── templates/
│   ├── chat.html          # Frontend HTML for the chatbot interface
├── static/
│   ├── style.css          # Styling for the web interface
├── app.py                 # Flask application
├── store_index.py         # Script to create and store FAISS index
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🖥️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/thEOneA/RAG_FAISS_Medical_Chatbot.git
   cd RAG_FAISS_Medical_Chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Llama 2 model:
   - Place the model file (e.g., `llama-2-7b-chat.ggmlv3.q4_0.bin`) in the `model/` directory.

5. Create a `.env` file:
   ```bash
   touch .env
   ```
   Add the following environment variables:
   ```
   PINECONE_API_KEY=<your_pinecone_api_key>  # Optional if using Pinecone
   ```

6. Build the FAISS index:
   ```bash
   python store_index.py
   ```

7. Run the Flask app:
   ```bash
   python app.py
   ```

---

## 🌐 Usage

1. Open your browser and navigate to:
   ```
   http://127.0.0.1:8080
   ```

2. Enter your medical query in the chatbox and get instant responses.

---

## 📝 Customization

- **Modify Prompts**: Edit `src/prompt.py` to customize the chatbot's behavior.
- **Add Documents**: Place new medical PDFs in the `data/` directory and rebuild the FAISS index using `store_index.py`.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.


