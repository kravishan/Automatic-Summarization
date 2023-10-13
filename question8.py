import os
import nltk
import requests
from rouge import Rouge
from lxml import html
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Initialize the ROUGE scorer
rouge = Rouge()

# Create a list of paper URLs
paper_urls = [
    "https://pubs.acs.org/doi/pdf/10.1021/cs200217t?casa_token=UXuYNyAG_OwAAAAA:y42dcrKlZJvCqmd5H5K5Xd9b2RKUE67ZKFTiLA1I1WTrAtBKtb8CDtG5U422Mq0oVeJo8oX4P5kJmw",
    "https://pubs.acs.org/doi/pdf/10.1021/cs200217t?casa_token=UXuYNyAG_OwAAAAA:y42dcrKlZJvCqmd5H5K5Xd9b2RKUE67ZKFTiLA1I1WTrAtBKtb8CDtG5U422Mq0oVeJo8oX4P5kJmw",
    # Add URLs for all the papers
]

# Initialize a list to store ROUGE scores for each paper
rouge_scores = []

# Loop through the list of paper URLs to download and process each paper
for paper_url in paper_urls:
    # Fetch the paper content using the requests library
    response = requests.get(paper_url)
    if response.status_code == 200:
        paper_content = response.content

        # Parse the paper content using the lxml parser
        soup = html.fromstring(paper_content)

        # Extract the paper's main content (e.g., the article body)
        main_content = soup.find("div", {"class": "article-body"})  # Modify to match your paper's HTML structure

        if main_content:
            # Create a parser and tokenizer for the paper content
            parser = HtmlParser(main_content, Tokenizer("english"))

            # Summarization with TextRank
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, 2)  # Set the number of sentences in the summary

            # Convert summaries to text
            summary_text = " ".join([str(sentence) for sentence in summary])

            # Define reference summaries (abstract and conclusion)
            abstract = "Your abstract here"
            conclusion = "Your conclusion here"

            # Calculate ROUGE scores for TextRank summary
            rouge_scores_textrank = rouge.get_scores(summary_text, abstract)

            # Store the ROUGE scores for this paper
            rouge_scores.append({
                "Paper URL": paper_url,
                "TextRank_ROUGE_1": rouge_scores_textrank[0]["rouge-1"]["f"],
                "TextRank_ROUGE_2": rouge_scores_textrank[0]["rouge-2"]["f"]
            })

        else:
            print(f"Paper content not found for URL: {paper_url}")

    else:
        print(f"Failed to fetch paper from URL: {paper_url}")

# Print the ROUGE scores for each paper
for score in rouge_scores:
    print(f"Paper URL: {score['Paper URL']}")
    print(f"TextRank_ROUGE_1: {score['TextRank_ROUGE_1']}")
    print(f"TextRank_ROUGE_2: {score['TextRank_ROUGE_2']}")
    print("\n")
