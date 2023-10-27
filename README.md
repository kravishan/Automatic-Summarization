# Project 29: Automatic Summarization 2

This repository contains a Python script that aids in the extraction of crucial keywords from structured HTML documents. The utility aims to identify significant keywords, which are often present in the titles, abstracts, and section/subsection headers of the document.

## Features

- Extracts the main title of the HTML document.
- Retrieves the abstract of the document.
- Gathers all the titles of sections and subsections present in the document.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Libraries: 
  - `beautifulsoup4`
  - `lxml`

You can install these libraries using:

```bash
pip install requirements.txt
```

## Usage

1. Clone this repository:
```bash
git clone https://github.com/kravishan/Automatic-Summarization.git
cd Automatic-Summarization
```

2. Run the script:
```bash
python3 task-1.py example.html
```

The script will output the main title, abstract, and all section/subsection titles present in the given HTML document.

For the provided `example.html` with the structure described earlier, the expected output will be:

   ```plaintext
   Title: Document Keyword Extraction
   Abstract: This study is among the first attempts to empirically investigate the adoption of mobile government by rural populations in developing economies.
   Section Titles: ['Introduction', 'Literature review', 'Research model and the development of hypotheses', 'Research methodology', 'Results', 'Conclusion']
   ```

---



   

