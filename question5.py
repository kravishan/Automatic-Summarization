import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

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

# Calculate semantic similarity between sentences using cosine similarity
def calculate_similarity(sentences, word_model):
    sentence_vectors = [
        np.mean([word_model[word] for word in sentence if word in word_model], axis=0)
        for sentence in sentences
    ]
    return cosine_similarity(sentence_vectors, sentence_vectors)

# Choose the top 10 diverse sentences based on the described approach
def choose_diverse_sentences(similarity_matrix):
    num_sentences = len(similarity_matrix)
    selected_indices = []  
    selected_indices.append(0)  

    # Select the remaining 9 sentences
    for _ in range(9):
        min_similarity = float("inf")
        selected_sentence = -1

        for i in range(num_sentences):
            if i in selected_indices:
                continue

            # Calculate similarity to the first selected sentence (S1)
            sim_to_S1 = similarity_matrix[i][selected_indices[0]]

            # Calculate similarity to the sentence that minimizes "Sim(Sp, S1) + Sim(Sp, Sj)"
            for j in selected_indices[1:]:
                sim_to_Sj = similarity_matrix[i][j]
                total_similarity = sim_to_S1 + sim_to_Sj

                if total_similarity < min_similarity:
                    min_similarity = total_similarity
                    selected_sentence = i

        selected_indices.append(selected_sentence)

    return selected_indices

# Load the Word2Vec model
word_model = KeyedVectors.load_word2vec_format("E:\Automatic-Summarization-\Models\GoogleNews-vectors-negative300.bin", binary=True)

# Example usage
document = """
In the context of online reviews, personal factors can influence how many reviews consumers read, the type of reviews they read, and how they interpret the reviews. For example, consumers with a high decision-making drive may read fewer reviews, but they may be more likely to focus on reviews that are written by experts or other knowledgeable consumers. Consumers who are more uncertain about the product may read more reviews, and they may be more likely to focus on reviews that discuss the pros and cons of the product in detail.
"""
cleaned_sentences = preprocess_document(document)
similarity_matrix = calculate_similarity(cleaned_sentences, word_model)
selected_indices = choose_diverse_sentences(similarity_matrix)

# Print the selected diverse sentences
for index in selected_indices:
    print(cleaned_sentences[index])
