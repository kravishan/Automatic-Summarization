from bs4 import BeautifulSoup

# Sample HTML document 
html_document = """
<html>
  <head>
    <title>Sample HTML Document</title>
  </head>
  <body>
    <h1>Main Title</h1>
    <p>This is the abstract of the document.</p>
    <section>
      <h2>Section 1: Introduction</h2>
      <p>Content of Section 1.</p>
    </section>
    <section>
      <h2>Section 2: Methodology</h2>
      <p>Content of Section 2.</p>
    </section>
  </body>
</html>
"""

# Parse the HTML document 
soup = BeautifulSoup(html_document, 'html.parser')

# Extract the words from the title, abstract, and titles of sections
title = soup.title.string if soup.title else None
abstract = soup.find('p').get_text() if soup.find('p') else None
section_titles = [section.h2.get_text() for section in soup.find_all('section')]

# Tokenize the text to get a list of words
def extract_words(text):
    if text:
        return text.split()
    return []

# Extract words from the title, abstract, and section titles
title_words = extract_words(title)
abstract_words = extract_words(abstract)
section_title_words = [extract_words(title) for title in section_titles]

# Print the results
print("Title Words:", title_words)
print("Abstract Words:", abstract_words)
print("Section Title Words:", section_title_words)
