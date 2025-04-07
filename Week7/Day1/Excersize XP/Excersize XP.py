
'''


Exercises XP


What You'll learn
Text preprocessing usage
Text Analysis technics
POS and NER tags
vectorization and word embeddings: Word2Vec


'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

Exercise 1: Exploring Text Preprocessing usage, NER and POS tags
1. Create a function preprocess_text() which will receive the data as argument and:

convert all the text in lower case and tokanize it
remove punctuation
apply a lemmatizer
return the preprocessed strings


important:
after creating each function, apply it in the dataset and print the result to check that is working properly

'''

# Download necessary NLTK data
import nltk
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
import string
import spacy

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(data):

    lemmatizer = WordNetLemmatizer()
    preprocessed_data = []

    for review in data['Review']:
        # Convert to lowercase
        review = review.lower()
        # Tokenize
        tokens = word_tokenize(review)
        # Remove punctuation
        tokens = [word for word in tokens if word not in string.punctuation]
        # Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back to string
        preprocessed_data.append(' '.join(tokens))

    return preprocessed_data

# Our example data
data = {
    'Review': [
        'At McDonald\'s the food was ok and the service was bad.',
        'I would not recommend this Japanese restaurant to anyone.',
        'I loved this restaurant when I traveled to Thailand last summer.',
        'The menu of Loving has a wide variety of options.',
        'The staff was friendly and helpful at Google\'s employees restaurant.',
        'The ambiance at Bella Italia is amazing, and the pasta dishes are delicious.',
        'I had a terrible experience at Pizza Hut. The pizza was burnt, and the service was slow.',
        'The sushi at Sushi Express is always fresh and flavorful.',
        'The steakhouse on Main Street has a cozy atmosphere and excellent steaks.',
        'The dessert selection at Sweet Treats is to die for!'
    ]
}

preprocessed_reviews = preprocess_text(data)
print(preprocessed_reviews)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

2. Create a new dataset with the cleaned text

hint: keep two datasets: the raw data and the preprocessed data

'''

raw_data = data  # Keep the raw data
cleaned_data = {'Review': preprocessed_reviews}  # Create a new dataset with preprocessed reviews


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

3. Create a function perform_ner() that will receive the text as argument and perform NER tagging on it. Use spacy en_core_web_sm

hint: the function should return the entities text and label_ (example of _labels: ORG, GPE, DATE)

'''

# Load the spaCy model
import spacy
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# use scacy "en_core_web_sm" model for NER
nlp = spacy.load("en_core_web_sm")

def perform_ner(text):
    doc = nlp(text)  # doc = nlp("I visited Google in California last summer.")
    entities = [(ent.text, ent.label_) for ent in doc.ents]  # doc.ents = (Google, California, last summer)
    return entities

# Example usage
example_text = "I visited Google in California last summer."
entities = perform_ner(example_text)
print(entities)
# [('Google', 'ORG'), ('California', 'GPE'), ('last summer', 'DATE')]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

4. Create a function perform_pos_tagging() that will receive the text as argument and perform POS tagging on it.

hint: use nltk pos_tag method


'''

from nltk.tag import pos_tag
from gensim.models import Word2Vec

def perform_pos_tagging(text):
    tokens = word_tokenize(text)  # Tokenize the text
    pos_tags = pos_tag(tokens)  # Perform POS tagging
    return pos_tags

# Example usage
example_text = "I visited Google in California last summer."
pos_tags = perform_pos_tagging(example_text)
print(pos_tags)
# [('I', 'PRP'), ('visited', 'VBD'), ('Google', 'NNP'), ('in', 'IN'), ('California', 'NNP'), ('last', 'JJ'), ('summer', 'NN'), ('.', '.')]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

5. Let's apply the functions on the dataset, and analyse the outputs, applying it on the preprocessed data and on the raw data.
hint: to understand the meaning of the different POS tags you can use:

nltk.download('tagsets')
nltk.help.upenn_tagset('NN')

'''


# Download tagsets for understanding POS tags
nltk.download('tagsets')

# Analyze POS tags meaning
nltk.help.upenn_tagset('NN')

# Apply NER and POS tagging on raw and preprocessed data

# Raw data analysis
print("Raw Data Analysis:")
for review in raw_data['Review']:
    print(f"Review: {review}")
    print(f"NER: {perform_ner(review)}")
    print(f"POS Tags: {perform_pos_tagging(review)}")
    print()

# Preprocessed data analysis
print("Preprocessed Data Analysis:")
for review in cleaned_data['Review']:
    print(f"Review: {review}")
    print(f"NER: {perform_ner(review)}")
    print(f"POS Tags: {perform_pos_tagging(review)}")
    print()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


Exercise 2: Plotting the word embeddings


'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

1. Create the word embeddings using Word2Vec model to vectorize the text.

hint: use the preprocessed and tokenized dataset and use Word2Vec model from gensim.models

Print the dimensions of the Word2Vec object and analyse it.
What are the vector dimensions ?
What does it mean ?

'''

import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# Tokenize the preprocessed data
tokenized_reviews = [review.split() for review in cleaned_data['Review']]

# Create Word2Vec model
# vector_size = represents the dimensionality of the word vectors, (embeddings) for words learned by the model
word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
# min_count = ignores all words with total frequency lower than this

# word2vec_model.wv.index_to_key is a list of words in the vocabulary - It lists all the words (or tokens) the Word2Vec model has learned
# The words are sorted by their frequency

# Print the dimensions of the Word2Vec object
print(f"Vocabulary Size: {len(word2vec_model.wv.index_to_key)}")
print(f"Vector Dimensions: {word2vec_model.vector_size}")
# The vector dimensions are 100


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

2.
Create a function plot_word_embeddings() that receives the word2vec object as argument and plots the embeddings dimensions in a grided plot.
Use a scatter plot.
Loop through the words and use annotate() method to add text labels to each point on the scatter plot.
Finally call this function to see the plots and analyse it:
Are the related words close to each other?  What can be the possible reasons for this output?

'''


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_word_embeddings(word2vec_model):
    
    # Extract word vectors and corresponding words
    words = list(word2vec_model.wv.index_to_key)
    word_vectors = word2vec_model.wv[words]

    # Reduce dimensions using PCA for visualization.
    pca = PCA(n_components=2)  # Reduce to 2 dimensions so we can plot it
    reduced_vectors = pca.fit_transform(word_vectors)

    # Plot the embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # Annotate each point with the corresponding word
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9, alpha=0.8)

    plt.title("Word Embeddings Visualization")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True)
    plt.show()

# Call the function to plot word embeddings
plot_word_embeddings(word2vec_model)

# Are the related words close to each other?  # What can be the possible reasons for this output?
# Actually, I don't see that related words are close to each other in the plot.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

3. To enhance the analysis, you can:

Experiment with different preprocessing techniques.
Fine-tune the Word2Vec model parameters.
Explore advanced visualization techniques for word embeddings.



'''

# Fine-tune Word2Vec model parameters
fine_tuned_word2vec_model = Word2Vec(
    sentences=tokenized_reviews,
    vector_size=150,  # Increased vector size
    window=3,         # Reduced context window size
    min_count=2,      # Ignore words with frequency less than 2
    workers=4,
    sg=1              # Use skip-gram model
)

# Advanced visualization techniques for word embeddings
# We will reduce dimensions using t-SNE (and not using PCA)

def tsne_word_embeddings(word2vec_model):

    # Extract word vectors and corresponding words
    words = list(word2vec_model.wv.index_to_key)
    word_vectors = word2vec_model.wv[words]

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
    reduced_vectors = tsne.fit_transform(word_vectors)

    # Plot the embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # Annotate each point with the corresponding word
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9, alpha=0.8)

    plt.title("Word Embeddings Visualization (t-SNE)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.show()

# Call the t-SNE visualization function
tsne_word_embeddings(fine_tuned_word2vec_model)


