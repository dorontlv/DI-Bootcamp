'''


Text Analysis of books ...


Daily Challenge : Text Analysis of books using word cloud


What we will learn
Text preprocessing
Text Analysis
Bag of words (BoW) method
TF-IDF


Important
Create a virtual enviroment to the NLP course and work always on it.


'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

Text preprocessing
For this exercises we will be using NLTK and spaCy

The corpus will be the Lewis Carrol books:

Alice's Adventures in Wonderland
https://www.gutenberg.org/cache/epub/11/pg11.txt

THROUGH THE LOOKING-GLASS And What Alice Found There
https://www.gutenberg.org/cache/epub/12/pg12.txt

A Tangled Tale
https://www.gutenberg.org/cache/epub/29042/pg29042.txt

Using requests to access the contents online, create a function load_texts().This function should recive a list of urls, load them, clean non-words using regular expressions and append the cleaned text to the corpus that will be returned.

print the first 200 characteres of each text.
Are there parts of the text that are not relevant to the analysis? If so, you need to remove them.
hint:* you can use slicing to start and stop the text where you need (ignoring autoral credits in the begining and end) looking for the following phrases:
' START'
'*** END'

tokenize the text and print the first 150 tokens of each book

remove stopwords using NLTK.
Check that they were removed using count() and look for some of the stop words like: 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', etc.

Using PorterStemmer(), print the first 50 stemmed tokens
Using spaCy pre-trained model 'en_core_web_sm' to load and print the first 50 lemmatized tokens. Hint: in spaCy the lemmatized token can be accessed as attribute.
Analyse the difference between the stemmed and lemmatized tokens. What is different and why?
using NLTK, identify POS tags on each text.
using NLTK identify all the entities of each text


'''


import requests
import re  # regular expressions
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


# get the texts from the internet
def load_texts(urls):

    corpus = []
    for url in urls:
        response = requests.get(url)  # get the texts from the internet
        text = response.text

        # Extract relevant part of the text - START to END
        start_idx = text.find('START') + len('START')
        end_idx = text.find('*** END')
        text = text[start_idx:end_idx]

        # Clean non-words using regex
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        corpus.append(text)

    return corpus


# URLs of the books
urls = [
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "https://www.gutenberg.org/cache/epub/12/pg12.txt",
    "https://www.gutenberg.org/cache/epub/29042/pg29042.txt"
]


# Load and preprocess texts
texts = load_texts(urls)

# Print first 200 characters of each text
for i, text in enumerate(texts):
    print(f"First 200 characters of book {i + 1}:\n{text[:200]}\n")

# Tokenize texts
tokenized_texts = [nltk.word_tokenize(text) for text in texts]
for i, tokens in enumerate(tokenized_texts):
    print(f"First 150 tokens of book {i + 1}:\n{tokens[:150]}\n")

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_texts = [[word for word in tokens if word.lower() not in stop_words] for tokens in tokenized_texts]
for i, filtered in enumerate(filtered_texts):
    print(f"Stopwords removed from book {i + 1}: Example 'we' count: {filtered.count('we')}\n")

# Stemming filtered_texts using PorterStemmer
stemmer = PorterStemmer()
stemmed_texts = [[stemmer.stem(word) for word in tokens] for tokens in filtered_texts]
for i, stemmed in enumerate(stemmed_texts):
    print(f"First 50 stemmed tokens of book {i + 1}:\n{stemmed[:50]}\n")

# Lemmatization filtered_texts using spaCy
nlp = spacy.load('en_core_web_sm')
lemmatized_texts = [[token.lemma_ for token in nlp(" ".join(tokens))] for tokens in filtered_texts]
for i, lemmatized in enumerate(lemmatized_texts):
    print(f"First 50 lemmatized tokens of book {i + 1}:\n{lemmatized[:50]}\n")

# POS tagging filtered_texts using NLTK
for i, tokens in enumerate(filtered_texts):
    pos_tags = nltk.pos_tag(tokens)
    print(f"POS tags of book {i + 1} (first 50):\n{pos_tags[:50]}\n")

# Entity recognition using NLTK
for i, tokens in enumerate(filtered_texts):
    entities = nltk.ne_chunk(nltk.pos_tag(tokens))
    print(f"Entities of book {i + 1} (first 50):\n{list(entities)[:50]}\n")




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Analysing the text
using wordcloud and matplotlib, display a word cloud of each book.
The output will look like this (maybe not exactly):

'''

# pip install wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate and display word clouds for each book
# it will be best to use the raw text, because the stemmed and lemmatized texts are not the original words.

for i, text in enumerate(texts):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Book {i + 1}")
    plt.show()


'''
Use BoW method to check the five most frequent words in all the books
hint: What will be the best text from the preprocess step? (raw text, stemmed, lemmatized, etc)?

print the BoW and identify the numbers: What is the document number? What is the index and what is how many times the word was found?
Display a pie plot of the 5 most frequent words in the text. Add the word and its frequence as labels.
Analyse the outputs: Are those words informative? Are they insightful or expected?

'''

# Using lemmatized words instead of the original raw text is good when counting the most common words in a large document, because lemmatization helps normalize word forms.

from collections import Counter

# Use lemmatized texts for BoW analysis
for i, lemmatized in enumerate(lemmatized_texts):
    # Count word frequencies
    word_counts = Counter(lemmatized)
    most_common = word_counts.most_common(5)  # most common 5 words
    
    # Print BoW details
    print(f"Book {i + 1} - BoW Most Common Words:")
    for word, count in most_common:
        print(f"Word: '{word}', Frequency: {count}")
    print()
    
    # Create pie chart
    words, counts = zip(*most_common)  # break into words and counts
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=[f"{word} ({count})" for word, count in most_common], autopct='%1.1f%%', startangle=140)
    plt.title(f"Top 5 Words in Book {i + 1}")
    plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

Solving the frequency problem using TF-IDF
When we create a BoW out from some text, all the words are treated equaly as importants. Like "Alice" or "say" in a fantasy book about Alice. We expected those words to be repeated all over the book, making them not so informative to us.

The solution for this problem would be to consider the frequency relative to the corpus. In this case, if there is a word in a document that doesn't appears much in the other documents, it is likely meaningful and should be considered more important. And the same way in the oposite: A word that is repeated a lot in all the documents will be considered less important.

That's the perfect situation to use TF-IDF (Term Frequency-Inverse Document Frequency)

1. Create another BoW, now using TF-IDF as vectorizer.
hint: You need to pass min_df=1, max_df=2 as arguments to the TfidfVectorizer() function, because we are using a small dataset.
2. Create again the pie plots with the new 5 most relevant words from each document.

'''

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=2, stop_words='english')

# Fit and transform the texts
# We need to join the tokens back into strings for TF-IDF
# fit_transform() gets a list of all 3 texts.
# Note: We are using the filtered texts (without stopwords) for TF-IDF analysis.
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(tokens) for tokens in filtered_texts])

# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Analyze top 5 words for each document
for i in range(len(filtered_texts)):
    
    # Get TF-IDF scores for the current document
    # take the i-th document (book).
    # convert toarray() and to a 1D array.
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    
    # Get top 5 words and their scores
    top_indices = tfidf_scores.argsort()[-5:][::-1]  # sorting in descending order, then taking the last 5
    top_words = [(feature_names[idx], tfidf_scores[idx]) for idx in top_indices]
    
    # Print TF-IDF details
    print(f"Book {i + 1} - TF-IDF Most Relevant Words:")
    for word, score in top_words:
        print(f"Word: '{word}', TF-IDF Score: {score:.4f}")
    print()
    
    # Create pie chart - put all the data into a pie chart
    words, scores = zip(*top_words)  # unpacking into the zip function
    plt.figure(figsize=(6, 6))
    plt.pie(scores, labels=[f"{word} ({score:.2f})" for word, score in top_words], autopct='%1.1f%%', startangle=140)
    plt.title(f"Top 5 TF-IDF Words in Book {i + 1}")
    plt.show()





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

Asset :
Alice's Adventures in Wonderland

THROUGH THE LOOKING-GLASS And What Alice Found There

A Tangled Tale



'''