# import the required libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
from collections import Counter 
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize 

def generate_summary_F_Based(text, n):

    # Tokenize the text into individual sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into individual words and remove stopwords
    stop_words = set(stopwords.words('english'))


    words = []
    for word in word_tokenize(text):
        if word.lower() not in stop_words and word.isalnum():
            words.append(word.lower())


    # Compute the frequency of each word
    word_freq = Counter(words)

    # empty dictionary to store the scores for each sentence
    sentence_scores = {}

    for sentence in sentences:
        sentence_words = []
        for word in word_tokenize(sentence):
            if word.lower() not in stop_words and word.isalnum():
                sentence_words.append(word.lower())
        sentence_score = 0
        for word in sentence_words:
            sentence_score += word_freq[word]
            sentence_scores[sentence] = sentence_score
        # if len(sentence_words) < 20:
        #     sentence_scores[sentence] = sentence_score


    # Select the top n sentences with the highest scores
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
    summary = ' '.join(summary_sentences)

    return summary