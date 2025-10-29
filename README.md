# 🧠 RAG_Model — End-to-End Retrieval-Augmented Generation (RAG) using ObjectBox + LangChain

This project implements an **End-to-End Retrieval-Augmented Generation (RAG)** pipeline combining:
- **LangChain** — for connecting retrieval and generation workflows.
- **ObjectBox Vector Database** — for fast and persistent embedding storage.
- **HuggingFace Embeddings** — for converting text into numerical vectors.
- **Groq LLM (Llama 3)** — for generating intelligent, context-aware answers.

It allows you to upload your **own documents (e.g., research papers)**, automatically index them using vector embeddings, and query them in natural language to receive accurate answers based on the document content.

---

## 📘 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [How to Use the Application](#-how-to-use-the-application)
- [Technologies Used](#-technologies-used)
- [Code Explanation](#-code-explanation)
- [Demo Screenshots](#-demo-screenshots)
- [Example Queries](#-example-queries)
- [Troubleshooting Guide](#-troubleshooting-guide)
- [License](#-license)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)
- [Summary](#-summary)
- [Next Steps](#-next-steps)

---

## 🚀 Project Overview

Retrieval-Augmented Generation (RAG) combines **information retrieval** and **language generation**.

Instead of relying purely on a language model’s internal knowledge, RAG retrieves **relevant passages from your uploaded documents** and uses them as additional context for response generation.

---

## ⚙️ Architecture

### 🔹 RAG Pipeline Workflow

1. **Document Ingestion**
   - Loads your PDF documents from a specified folder.

2. **Text Splitting**
   - Breaks documents into overlapping text chunks using LangChain’s `RecursiveCharacterTextSplitter`.

3. **Embedding Creation**
   - Converts each text chunk into a numerical vector using HuggingFace’s `BGE-small-en-v1.5` model.

4. **Vector Storage**
   - Stores embeddings persistently in **ObjectBox Vector Database** for efficient retrieval.

5. **Retrieval**
   - Retrieves top similar document chunks based on your query.

6. **Generation**
   - Passes retrieved context to Groq’s **Llama 3** model, which generates a precise, context-grounded answer.

---

## 📂 Directory Structure

```plaintext
End-to-End-RAG-Project-using-ObjectBox-and-LangChain/
│
├── app/
│   ├── app.py                # Main Streamlit application
│   ├── utils.py              # Helper functions for embeddings & LLM
│   ├── config.py             # Loads environment variables (.env)
│
├── research-papers/          # Folder containing your research papers (PDFs)
├── objectbox/                # ObjectBox database files (auto-created)
│
├── requirements.txt          # Python dependencies
├── .env                      # API keys for HuggingFace and Groq
├── .gitignore                # Excludes sensitive files and venv
└── README.md                 # This documentation file
```

## Libraries Used
 - langchain==0.1.20
 - langchain-community==0.0.38
 - langchain-core==0.1.52
 - langchain-groq==0.1.3
 - langchain-objectbox
 - python-dotenv==1.0.1
 - pypdf==4.2.0

## Installation
 1. Prerequisites
    - Git
    - Command line familiarity
 2. Clone the Repository: `git clone https://github.com/NebeyouMusie/End-to-End-RAG-Project-using-ObjectBox-and-LangChain.git`
 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`
 4. Navigate to the projects directory `cd ./End-to-End-RAG-Project-using-ObjectBox-and-LangChain` using your terminal
 5. Install Libraries: `pip install -r requirements.txt`
 6. Navigate to the app directory `cd ./app` using your terminal 
 7. run `streamlit run app.py`
 8. open the link displayed in the terminal on your preferred browser
 9. As I have already embedded the documents you don't need to click on the `Embedd Documents` button/ But, if it's not working then you need to click on the `Embedd Documents` button and wait until the documnets are processed
 10. Enter your question from the PDFs found in the `us-census-data` directory




   
## Contact
 - LinkedIn: [Nebeyou Musie](https://www.linkedin.com/in/nebeyou-musie)
 - Gmail: nebeyoumusie@gmail.com
 - Telegram: [Nebeyou Musie](https://t.me/NebeyouMusie)


