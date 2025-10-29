# 🧠 RAG_Model — End-to-End Retrieval-Augmented Generation (RAG) using ObjectBox + LangChain

This project implements an **End-to-End Retrieval-Augmented Generation (RAG)** pipeline combining:
- **LangChain** — for connecting retrieval and generation workflows.
- **ObjectBox Vector Database** — for fast and persistent embedding storage.
- **HuggingFace Embeddings** — for converting text into numerical vectors.
- **Groq LLM (Llama 3)** — for generating intelligent, context-aware answers.

It allows you to upload your **own documents (e.g., research papers)**, automatically index them using vector embeddings, and query them in natural language to receive accurate answers based on the document content.

---

## 🔗 Demo

🎥 **Live Demo:** [Click Here to View the App](https://your-demo-link-here.com)  
*(Replace with your Streamlit Cloud / HuggingFace Spaces / server link when deployed)*

---

## 📸 Screenshots

| 🏠 Home Interface | 💬 Query & Answer View |
|-------------------|-----------------------|
| ![Home Screenshot](screenshots/home_page.png) | ![Query Screenshot](screenshots/query_result.png) |

> 📁 Place your screenshots in a folder named `screenshots/`:
> ```bash
> mkdir screenshots
> mv ~/Downloads/home_page.png screenshots/
> mv ~/Downloads/query_result.png screenshots/
> ```



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
 2. Clone the Repository: `git clone https://github.com/shivaprogrammer/RAG_Model.git`
 3. Create and Activate Virtual Environment (Recommended)
    - `python -m venv venv`
    - `source venv/bin/activate`

 4. Install Libraries: `pip install -r requirements.txt`
 5. Navigate to the app directory `cd ./app` using your terminal 
 6. run `streamlit run app.py`
 7. open the link displayed in the terminal on your preferred browser
 8. As I have already embedded the documents you don't need to click on the `Embedd Documents` button/ But, if it's not working then you need to click on the `Embedd Documents` button and wait until the documnets are processed
 9. Enter your question from the PDFs found in the `research_papers` directory




   



