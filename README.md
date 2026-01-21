# 📄 Smart Document Indexer with Gemini & PostgreSQL

This project is a modular Python-based pipeline designed to ingest documents (PDF, DOCX), split them into chunks using various strategies, generate vector embeddings using **Google Gemini API**, and store them in a **PostgreSQL** database.

It can serve as the backend foundation for RAG (Retrieval-Augmented Generation) systems, allowing for efficient semantic search over large text corpuses.

## 🚀 Features

* **Multi-Format Support:** Handles `.pdf` and `.docx` files seamlessly.
* **Smart Chunking Strategies:**
    * `fixed`: Splits by character count with overlap.
    * `sentence`: Uses NLTK to respect sentence boundaries.
    * `paragraph`: Intelligent splitting based on document structure (supports both clean DOCX and messy PDFs).
* **Vector Embeddings:** Integration with Google Gemini (`embedding-001`) to generate high-quality vector representations.
* **Robust Error Handling:** Includes rate-limiting (handling API 429 errors) and automatic retries.
* **Batch Processing:** Supports processing multiple files via a JSON configuration file.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/DanielShapira19/document-indexer]
```

### 2. Install Dependencies
Make sure you have Python 3.12 installed.

```bash
pip install -r requirements.txt
```


### 3. Database Setup (PostgreSQL) and Environment Configuration (.env)
Since this project runs locally, you need to have a local PostgreSQL instance ready.

1.  **Install PostgreSQL:** Download and install it from [postgresql.org](https://www.postgresql.org/download/).
2.  **Create a Database:** Open pgAdmin (or your terminal) and create a new empty database named `index_documents_db`.
    ```sql
    CREATE DATABASE index_documents_db;
    ```
3. Create a `.env` file in the root directory. This file keeps your secrets safe and should **not** be committed to GitHub.

    ```env
    # Google Gemini API Key (Get one at aistudio.google.com)
    GEMINI_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxx
    
    # PostgreSQL Connection String
    Format: postgresql://user:password@host:port/database_name
    ```
4. **Run the Setup Script:** The Python script will handle the table creation for you. Just run:
    ```bash
    python index_documents.py --setup-db
    ```

## How to Run

You can run the indexer in two modes: **Batch Mode** (Recommended) or **Single File Mode**.

### Option A: Batch Mode (JSON Config)
This allows you to define multiple files and strategies in a single `jobs.json` file.

**Sample `jobs.json`:**
```json
{
  "documents": [
    {
      "path": "test.pdf",
      "strategies": ["fixed"],
      "chunk_size": 500,
      "overlap": 50
    },
    {
      "path": "cv.docx",
      "strategies": ["sentence", "paragraph"]
    }
  ]
}
```
**Command to initialize DB and run batch:**
Use this command to create the table (if it doesn't exist) and process the files defined in the JSON.

```bash
python index_documents.py --config jobs.json --setup-db
```

### Option B: Single File Mode (CLI):
Use this command to create the table (if it doesn't exist) and process the files defined in the JSON.
```bash
python index_documents.py --file "test.pdf" --strategies fixed sentence --chunk-size 1000```
```


### 🔧 Configuration Reference

**`jobs.json` Parameters:**
* `documents`: A list of file objects to be processed.
* `path`: The relative or absolute path to the document (PDF or DOCX).
* `strategies`: The split methods to apply. Options:
    * `"fixed"`: Splits by character count.
    * `"sentence"`: Splits by natural sentence boundaries (NLTK).
    * `"paragraph"`: Splits by structural paragraphs.
* `chunk_size`: (For `fixed` strategy) The number of characters per chunk.
* `overlap`: (For `fixed` strategy) The number of overlapping characters between chunks to preserve context.

**Command Flags:**
* `--config`: Specifies the path to the JSON batch file.
* `--setup-db`: Initializes the PostgreSQL table (`document_vectors`) if it doesn't exist.



## 📸 Execution Example

Here is a demonstration of a local run processing `test.pdf` using 3 different strategies.

### 1. Console Output
The script logs the progress, including chunk generation and API rate-limit handling.

![Console Output](./screen_shots/console.png)

### 2. Log File (`indexing.log`)
A persistent log file is created to track errors and successful operations.

![Log File](./screen_shots/log_file.png)

### 3. Database Results (PostgreSQL)
The data is stored in the `document_vectors` table.

![Database Results](./screen_shots/db.png)

**Table Schema Explanation:**

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | SERIAL | Unique auto-incrementing ID. |
| `chunk_text` | TEXT | The actual text segment used for search. |
| `embedding` | FLOAT8[] | The vector representation (numbers) from Gemini. |
| `filename` | TEXT | Source file name (e.g., `test.pdf`). |
| `strategy_split` | TEXT | How the text was split (e.g., `paragraph`, `fixed`). |
| `created_at` | TIMESTAMP | Time of insertion. |



