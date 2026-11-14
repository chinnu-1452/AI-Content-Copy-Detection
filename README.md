# 🔍 AI-Based Content Copy Detection System

An AI-powered semantic content copy detection system that identifies similar passages between a suspicious document and reference documents using NLP, Sentence Transformers, and FAISS.


## 🚀 Live Demo

**Try the Application:** [AI-Based Content Copy Detection System](https://ai-content-copy-detection-uqkpqgsurejvmal6hzmwzc.streamlit.app/)

## 💻 Source Code

**GitHub Repository:** [AI-Content-Copy-Detection](https://github.com/chinnu-1452/AI-Content-Copy-Detection)

## 📸 Screenshots
![Results](assets/results.png)

## 📌 Project Overview

This project analyzes uploaded documents and retrieves semantically similar passages from reference documents. It can identify content that has been rewritten using different words but retains a similar meaning.

## 🎯 Problem Statement

Traditional keyword-based matching may fail to identify paraphrased content. This project uses semantic embeddings to find passages with similar meanings, even when the wording is different.

## ✨ Features

- Upload PDF, DOCX, and TXT documents.
- Extract and preprocess document text.
- Split documents into smaller text chunks.
- Generate semantic embeddings using Sentence Transformers.
- Search similar passages using FAISS.
- Display similarity scores and possible source documents.
- Download analysis reports in Markdown and CSV formats.

## 🛠️ Technologies Used

- Python
- Streamlit
- NLP
- Sentence Transformers
- Hugging Face
- FAISS
- PyPDF
- Python-docx
- Pandas
- NumPy

## ⚙️ Project Workflow

1. Upload reference documents.
2. Upload the suspicious document.
3. Extract text from the documents.
4. Clean and split text into chunks.
5. Generate embeddings using Sentence Transformers.
6. Store reference embeddings in a FAISS index.
7. Retrieve semantically similar passages.
8. Display matching results and similarity scores.

## 📂 Project Structure

```text
AI_Content_Copy_Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── reference_docs/
│   └── uploads/
│
└── src/
    ├── __init__.py
    ├── extract.py
    ├── preprocess.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retrieval.py
    ├── similarity.py
    └── report.py
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/chinnu-1452/AI-Content-Copy-Detection.git
cd AI-Content-Copy-Detection
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

## 📊 Output

The application displays:

- Matching passages
- Semantic similarity scores
- Possible source documents
- Downloadable Markdown and CSV reports

## ⚠️ Limitations

- Results depend on the reference documents provided.
- Scanned PDFs requiring OCR are not supported in the current version.
- Similarity scores indicate semantic resemblance, not proof of plagiarism or AI authorship.
- The current version rebuilds the FAISS index during each analysis.

## 🔮 Future Enhancements

- Persistent FAISS vector database.
- Improved paraphrase detection.
- LLM-based match explanations.
- PDF report generation.
- Evaluation using Precision, Recall, F1-score, and Recall@K.
- Deployment to a cloud platform.

## 👨‍💻 Author

**Talari Pranay**

GitHub: https://github.com/chinnu-1452

---

*Built with Python, NLP, Sentence Transformers, FAISS, and Streamlit.*
