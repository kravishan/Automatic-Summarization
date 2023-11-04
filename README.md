# Project 29: Automatic Summarization 2

This repository contains a React web app and a Python server for creating a summarizer. It also includes a large number of summarization methods. The file `Answers.ipynb` is our Google Colab document that contains answers to all questions up to question number 8. We have developed a web app to showcase the solution to question number 9.

## Demonstration Video

For a comprehensive demonstration of the application setup and usage, watch our demo video on YouTube: [App Demo Video](https://youtu.be/bewVOYbnSx4)

[![App Demonstration](http://img.youtube.com/vi/bewVOYbnSx4/0.jpg)](https://youtu.be/bewVOYbnSx4)

This video guide covers the entire process, from starting the Python Flask server using `python3 app.py`, to initiating the React frontend server with `npm start`. It also provides a step-by-step walkthrough on how to enter a paragraph for summarization, showcasing the various summaries generated by our algorithms as well as the 'golden summary' crafted by ChatGPT.


## Features

* Introduction to fianlp
* Text preprocessing
* Tokenization
* Lemmatization
* Part-of-speech tagging
* Dependency parsing
* Named entity recognition
* Sentiment analysis
* Extract the main title of the HTML document
* Retrieves the abstract of the document
* Gathers all the titles of sections and subsections present in the document

## Prerequisites

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

2. Add your OpenAI API key
* Rename .env_sample file to .env
* Add your API key there

4. Start API:
```bash
cd server
python app.py
```
4. Start WEB APP
```bash
cd client
npm start
```
## Screenshots 

![UI](https://github.com/kravishan/Automatic-Summarization/assets/as-screenshot.png)







   

