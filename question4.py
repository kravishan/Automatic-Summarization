import requests
import nltk
from rouge import Rouge

# Function to summarize a document using your approach
def your_summarization_function(document):
    # Replace this with your actual summarization logic
    # You may use any summarization techniques you prefer, like extractive or abstractive summarization

    # For example, here's a simple placeholder that just takes the first few sentences as a summary
    sentences = nltk.sent_tokenize(document)
    num_sentences_to_include = 3
    summary = " ".join(sentences[:num_sentences_to_include])
    return summary

# Function to load reference summaries for a document
def load_reference_summaries(document):
    # Replace this with code to load reference summaries (ground truth) for the document
    # The format of the reference summaries may vary based on your dataset
    # You need to provide the actual implementation to load reference summaries

    # Here's a placeholder with empty reference summaries
    return []

# Define the base URL of the Opinosis dataset on GitHub
base_url = "https://raw.githubusercontent.com/kavgan/opinosis-summarization/master/sample_data/"

# List of document filenames you want to download and process
document_filenames = ["bestwestern.txt", "garmin.txt", "kindle.txt", "toyota_camry.txt"]

# Initialize Rouge
rouge = Rouge()

# Iterate over each document in the dataset
for filename in document_filenames:
    # Build the URL for the specific document
    document_url = f"{base_url}{filename}"

    # Download the document
    response = requests.get(document_url)

    if response.status_code == 200:
        # Read the downloaded data as text
        document = response.text

        # Implement your summarization approach to get the summary
        summary = your_summarization_function(document)

        # Load reference summaries (ground truth) for the document
        reference_summaries = load_reference_summaries(document)

        # Calculate Rouge-2 and Rouge-3 scores
        scores = rouge.get_scores(summary, reference_summaries)

        # Record the scores
        print(f"Document: {filename}")
        for score in scores:
            rouge_2_score = score['rouge-2']['f']
            rouge_3_score = score['rouge-3']['f']
            print(f"Rouge-2 Score: {rouge_2_score}, Rouge-3 Score: {rouge_3_score}")
        print("Recorded the Rouge-2 and Rouge-3 evaluation scores.")
    else:
        print(f"Failed to download the document: {filename}")
