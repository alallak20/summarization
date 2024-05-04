import math

# Tokenize and create a vocabulary
# every sentence into words as a list & a vocabulary containing all words 
def tokenize_and_create_vocabulary(sentences):
    # Initialize an empty list to store tokenized documents
    tokenized_sentences = []
    # Initialize an empty set to store ALL unique words in the vocabulary
    vocabulary = set()

    # Loop through each sentence
    for sentence in sentences:
        # Tokenize the document into individual words and convert them to lowercase
        tokenized_sentence = sentence.lower().split()
        # Add the tokenized document to the list of tokenized documents
        tokenized_sentences.append(tokenized_sentence)
        # Add unique words from the tokenized document to the vocabulary
        for word in tokenized_sentence:
            vocabulary.add(word)

    # Return the list of tokenized documents and the vocabulary
    return tokenized_sentences, vocabulary

# Calculate TF (Term Frequency)
def calculate_tf(tokenized_sentences, vocabulary):
    # Initialize an empty list to store TF matrices for each document
    tf_matrix = []

    # Loop through each document
    for sentence in tokenized_sentences:
        # Initialize an empty dictionary to store TF values for each word in the vocabulary
        tf_sentences = {}
        # Calculate the length of the document
        sentences_length = len(sentence)

        # Loop through each word in the vocabulary
        for word in vocabulary:
            # Calculate the term frequency of the word in the document
            tf_sentences[word] = sentence.count(word) / sentences_length

        # Add the TF dictionary for the document to the TF matrix
        tf_matrix.append(tf_sentences)

    # Return the TF matrix
    return tf_matrix

# Calculate IDF (Inverse Document Frequency)
def calculate_idf(tokenized_documents, vocabulary):
    # Initialize an empty dictionary to store IDF values for each word in the vocabulary
    idf_dict = {}
    # Get the total number of documents
    num_docs = len(tokenized_documents)

    # Loop through each word in the vocabulary
    for word in vocabulary:
        # Count the number of documents containing the word
        count = 0
        for doc in tokenized_documents:
            if word in doc:
                count += 1

        # Calculate IDF for the word
        idf_dict[word] = math.log(num_docs / (count + 1))

    # Return the IDF dictionary
    return idf_dict

# Calculate TF-IDF
def calculate_tfidf(tf_matrix, idf_dict):
    # Initialize an empty list to store TF-IDF matrices for each document
    tfidf_matrix = []

    # Loop through each document's TF matrix
    for tf_doc in tf_matrix:
        # Initialize an empty dictionary to store TF-IDF values for each word
        tfidf_doc = {}

        # Loop through each word and its TF value in the TF matrix
        for word, tf in tf_doc.items():
            # Calculate TF-IDF for the word
            tfidf_doc[word] = tf * idf_dict[word]

        # Add the TF-IDF dictionary for the document to the TF-IDF matrix
        tfidf_matrix.append(tfidf_doc)

    # Return the TF-IDF matrix
    return tfidf_matrix

# Calculate cosine similarity manually
def calculate_cosine_similarity(tfidf_matrix):
    # Initialize an empty list to store cosine similarity scores for each sentence
    sentence_scores = []

    # Loop through each document's TF-IDF matrix (excluding the last one, which is the query document)
    for tfidf_doc in tfidf_matrix[:-1]:
        # Calculate the cosine similarity score for the document
        score = sum(tfidf_doc.values())
        # Add the score to the list of sentence scores
        sentence_scores.append(score)

    # Return the list of sentence scores
    return sentence_scores