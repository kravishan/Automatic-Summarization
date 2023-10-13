import spacy
from collections import Counter
import matplotlib.pyplot as plt
from spacy.lang.en.stop_words import STOP_WORDS

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

# Sample document 
document = """
This is a sample document with some words. It contains stopwords, person names like John Smith and organization names like Google. It is just a sample document for demonstration.
"""

# Process the document with SpaCy
doc = nlp(document)

# Create a list of tokens without stopwords and punctuation
filtered_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS and not token.is_punct]

# Generate a histogram of word frequency
word_freq = Counter(filtered_tokens)

# Print word frequencies
print("Word Frequencies:")
for word, freq in word_freq.items():
    print(f"{word}: {freq}")

# Create a histogram plot of word frequency
plt.bar(word_freq.keys(), word_freq.values())
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.show()

# Identify person-named and organization-named entities
person_entities = [entity.text for entity in doc.ents if entity.label_ == "PERSON"]
organization_entities = [entity.text for entity in doc.ents if entity.label_ == "ORG"]

# Print named entities
print("\nPerson-named entities:")
for person in person_entities:
    print(person)

print("\nOrganization-named entities:")
for organization in organization_entities:
    print(organization)
