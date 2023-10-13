import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import RAKE

# Tokenize and preprocess the document
def preprocess_document(document):
    sentences = sent_tokenize(document)
    stop_words = set(stopwords.words("english"))
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    cleaned_sentences = [
        [word for word in tokens if word.isalnum() and word not in stop_words]
        for tokens in tokenized_sentences
    ]
    return cleaned_sentences

# Extract keywords using RAKE
def extract_keywords(text):
    rake = RAKE.Rake(RAKE.SmartStopList())
    keywords = rake.run(text)
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return [keyword[0] for keyword in keywords[:10]]

# Select sentences based on keywords
def select_sentences_by_keywords(sentences, top_keywords):
    selected_sentences = []

    for sentence in sentences:
        relevance_score = sum(1 for keyword in top_keywords if keyword in sentence)
        selected_sentences.append((sentence, relevance_score))

    selected_sentences = sorted(selected_sentences, key=lambda x: x[1], reverse=True)
    return [sentence for sentence, _ in selected_sentences[:10]]

# Example usage
document = """
In the context of online reviews, personal factors can influence how many reviews consumers read, the type of reviews they read, and how they interpret the reviews. For example, consumers with a high decision-making drive may read fewer reviews, but they may be more likely to focus on reviews that are written by experts or other knowledgeable consumers. Consumers who are more uncertain about the product may read more reviews, and they may be more likely to focus on reviews that discuss the pros and cons of the product in detail.
"""

cleaned_sentences = preprocess_document(document)
top_keywords = extract_keywords(document)
selected_sentences = select_sentences_by_keywords(cleaned_sentences, top_keywords)

# Print the selected sentences
for sentence in selected_sentences:
    print(sentence)
