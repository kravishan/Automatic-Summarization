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
The Industrial Revolution was a period of significant technological, economic, and social change that began in Britain in the late 18th century and spread to other parts of the world. It marked a major turning point in history as traditional agrarian economies transitioned to industrial and manufacturing-based societies.

During this period, there were remarkable innovations in various fields, including the mechanization of agriculture, the development of steam engines, and the growth of factories. These changes led to increased production and improved transportation systems, making goods more affordable and accessible.

The Industrial Revolution also had a profound impact on the labor force. With the rise of factories, many people moved from rural areas to urban centers in search of employment. This shift resulted in the growth of cities and a significant change in living conditions.

Technological advancements, such as the invention of the spinning jenny and the power loom, revolutionized the textile industry. Iron and steel production saw significant developments, as did the construction of railways and ships, enabling faster and more efficient transportation of goods and people.

The spread of the Industrial Revolution to other parts of the world, including the United States and continental Europe, brought about further industrialization. These changes paved the way for the modern world and had lasting effects on society, politics, and the global economy.

In summary, the Industrial Revolution was a period of profound change that marked the transition from agrarian economies to industrialized, manufacturing-based societies. It led to technological advancements, urbanization, and significant improvements in production and transportation. The effects of the Industrial Revolution are still felt today, shaping the world we live in.

"""
language = "english" 

summaries = summarize_with_various_approaches(document, language)

for approach, summary in summaries.items():
    print(f"Summary using {approach}:\n")
    for sentence in summary:
        print(sentence)
    print("\n")
