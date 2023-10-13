from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# Import NLTK and download the tokenizer model
import nltk
nltk.download("punkt")

# Function to summarize the document using different summarization approaches
def summarize_with_various_approaches(document, language):
    # Create a parser for the document
    parser = PlaintextParser.from_string(document, Tokenizer(language))
    
    summarizers = {
        "LexRank": LexRankSummarizer(),
        "LSA": LsaSummarizer(),
        "KL": KLSummarizer(),
        "TextRank": TextRankSummarizer(),
    }

    summaries = {}

    for approach, summarizer in summarizers.items():
        summary = summarizer(parser.document, 10)  
        summaries[approach] = summary

    return summaries

# Input the document you want to summarize
document = """
In the context of online reviews, personal factors can influence how many reviews consumers read, the type of reviews they read, and how they interpret the reviews. For example, consumers with a high decision-making drive may read fewer reviews, but they may be more likely to focus on reviews that are written by experts or other knowledgeable consumers. Consumers who are more uncertain about the product may read more reviews, and they may be more likely to focus on reviews that discuss the pros and cons of the product in detail.
"""
language = "english" 

summaries = summarize_with_various_approaches(document, language)

for approach, summary in summaries.items():
    print(f"Summary using {approach}:\n")
    for sentence in summary:
        print(sentence)
    print("\n")
