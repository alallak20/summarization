from nltk.tokenize import sent_tokenize
from heapq import nlargest
from TF_IDF.services import *



# Function to generate the summary manually(TF-IDF)
def generate_summary_manual_TF_IDF(text, n):
    # Tokenize the text into individual sentences
    sentences = sent_tokenize(text)

    # Tokenize and create a vocabulary
    tokenized_documents, vocabulary = tokenize_and_create_vocabulary(sentences)

    # Calculate TF
    tf_matrix = calculate_tf(tokenized_documents, vocabulary)

    # Calculate IDF
    idf_dict = calculate_idf(tokenized_documents, vocabulary)

    # Calculate TF-IDF
    tfidf_matrix = calculate_tfidf(tf_matrix, idf_dict)

    # Compute the cosine similarity between each sentence and the document
    sentence_scores = calculate_cosine_similarity(tfidf_matrix)

    # Select the top n sentences with the highest scores
    summary_sentences = nlargest(n, range(len(sentence_scores)), key=sentence_scores.__getitem__)

    # Retrieve the text of the selected summary sentences
    summary_sentences_text = []
    for i in sorted(summary_sentences):
        summary_sentences_text.append(sentences[i])
    summary_tfidf = ' '.join(summary_sentences_text)

    # Return the generated summary
    return summary_tfidf