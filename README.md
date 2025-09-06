# PDFChat

**PDFChat** is a simple and efficient tool that enables interactive conversations with PDF documents.  
This project was developed as part of an **internship**, and it was designed to run **completely offline** based on company requirements and strict privacy considerations.  

By leveraging embeddings, vector databases, and large language models (LLMs), it allows you to query, understand, and summarize PDF content securely without requiring cloud services.

---

## 🚀 Features

- **Offline PDF Chatting**  
  - Upload one or more PDF documents and ask questions about their content entirely offline.

- **Privacy-Focused**  
  - No external API calls; all processing is handled locally to comply with company policies.

- **Vector Storage**  
  - Embeds documents into ChromaDB for efficient retrieval.

- **LLM Integration**  
  - Uses **Ollama** with **LLaMA 3** as the language model backend, running locally.

- **Streamlit UI**  
  - Intuitive and interactive web interface for seamless usage.

---

## 📦 Requirements

- Python 3.8+
- `pip`

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Aditya-gautam21/pdfchat.git
   cd pdfchat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following:
   ```env
   DEPLOYED=False
   ```

4. Ensure Ollama is set up and LLaMA 3 is available locally:
   ```bash
   ollama run llama3
   ```

---

## ▶️ Usage Instructions

### Run the App
```bash
streamlit run app.py
```

### In the Web Interface

- Upload PDFs
- Ask questions related to the documents
- Get responses powered by embeddings + offline LLM

---

## 📂 Project Structure

```
pdfchat/
├── app.py                # Main Streamlit interface
├── embeddings.py         # Embedding functions for PDFs
├── userinput.py          # Streamlit user input components
├── vectorstore.py        # ChromaDB integration
├── requirements.txt      # Python dependencies
└── .env                  # Config file (not committed)
```

---

## 🌐 Configuration & Deployment

- Edit `.env` for deployment flags (e.g., `DEPLOYED=True` for production).  
- Ollama must be running locally with the selected LLM (`llama3`).  
- Embeddings and vectorstore are managed completely offline.  

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -m "Add awesome feature"`)  
4. Push and open a Pull Request  

---

## 📜 License

This project is open-source under the **MIT License**. Feel free to use, modify, and distribute!

---

## 🙏 Acknowledgements

- **Streamlit** – Web interface  
- **LangChain** – Orchestration framework  
- **ChromaDB** – Vector database  
- **Ollama & LLaMA 3** – Offline language model backend

---

## ⭐ About

PDFChat was created as an **internship project** to meet company requirements of **offline, privacy-first document interaction**.  
It provides a secure interface to turn static PDF documents into interactive knowledge bases using embeddings and LLMs, without depending on the internet or external services.
