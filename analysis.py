#Importing the files
import numpy as np
import pandas as pd
import math
from sklearn import model_selection

import tensorflow as tf
import tensorflow_hub as hub

from enhancements import Enhanced

SENTIMENT_LABELS = [
    "negative", "somewhat negative", "neutral", "somewhat positive", "positive"
]

enhancement = Enhanced()

# Add a column with readable values representing the sentiment.
def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
    range(5), SENTIMENT_LABELS)

# Create a DataFrame.
def load_data(file):
    return pd.read_csv(file, sep="\t", header=0)


# The data does not come with a validation set so we'll create one from the
# training set.
def get_data(train_file, test_file, validation_set_ratio=0.1):
    train_df = load_data(train_file)
    test_df = load_data(test_file)

    # Add a human readable label.
    add_readable_labels_column(train_df, "Sentiment")

    # We split by sentence ids, because we don't want to have phrases belonging
    # to the same sentence in both training and validation set.
    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
    test_size=validation_set_ratio,
    random_state=0)

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)]
    train_df = train_df[train_df["SentenceId"].isin(train_indices)]
    print("Split the training data into %d training and %d validation examples." %
            (len(train_df), len(validation_df)))

    return train_df, validation_df, test_df

def find_loglikelihood(loglikelihood_w_given, word, senti_dict, senti_label, V_senti_freq, V_total):
    k = enhancement.add_k_smoothing("add_k")
    loglikelihood_w_given[word] = dict()
    for senti_ind in range(len(senti_dict)):
        if word in senti_dict[senti_ind]:
            count_w = senti_dict[senti_ind][word]
        else:
            count_w = 0
        loglikelihood_w_given[word][senti_label[senti_ind]] = math.log(count_w + k) - math.log(V_senti_freq[senti_ind] + k*V_total)

def learn(train_df):
    senti_counts = np.zeros(5)
    senti_dict = [dict(),dict(),dict(),dict(),dict()] #key=token, value=frequency
    senti_label = ["negative","swnegative","neutral","swpositive","positive"]
    V_senti_freq = [0,0,0,0,0]
    V = dict()

    for _, row in train_df.iterrows():
        sentiment_index = row['Sentiment']
        senti_counts[sentiment_index] += 1
        token_list = []
        token_list  = row['Phrase'].rstrip("\n").split(" ")
        token_list = enhancement.remove_stopwords(token_list)
        token_list = enhancement.lower_case(token_list)
        token_list = enhancement.remove_punctuation(token_list)
        token_list = enhancement.stemming(token_list)
        token_list = enhancement.lemmatize(token_list)


        for token in token_list:
            if token in senti_dict[sentiment_index]:
                senti_dict[sentiment_index][token] += 1
            else:
                senti_dict[sentiment_index][token] = 1
            V_senti_freq[sentiment_index] += 1
            if token in V:
                if row['SentimentLabel'] in V[token]:
                    V[token]['SentimentLabel'] += 1
                else:
                    V[token]['SentimentLabel'] = 1
            else:
                V[token] = dict()
                V[token]['SentimentLabel'] = 1

    #Calculating P(c)
    total_senti_count = sum(senti_counts)
    p_senti = np.divide(senti_counts,total_senti_count)
    logprior_senti = np.log(p_senti)

    #Calculating P(wi|c)
    V_total = len(V)
    loglikelihood_w_given = dict()

    for word in V:
        find_loglikelihood(loglikelihood_w_given, word, senti_dict, senti_label, V_senti_freq, V_total)

    print("Learnt!")
    print("Log_priors:",logprior_senti)
    return senti_label, V, V_senti_freq, logprior_senti, loglikelihood_w_given
    
def classify(validation_df, senti_label, V, V_senti_freq, logprior_senti, loglikelihood_w_given):
    print("In classify")
    phraseId = []
    sentiment = []
    for _, row in validation_df.iterrows():
        sum_senti = logprior_senti
        word_list = []
        word_list  = row['Phrase'].rstrip("\n").split(" ")
        for word in word_list:
                if word in V:
                    for senti_ind in range(len(senti_label)):
                        sum_senti[senti_ind] = sum_senti[senti_ind] + loglikelihood_w_given[word][senti_label[senti_ind]]
        phraseId.append(row['PhraseId'])
        sentiment.append(np.argmax(sum_senti))

    # df = pd.DataFrame(data=classified_dict)
    print("Classified!")
    return phraseId, sentiment

def verify(phraseId, sentiment, validation_df):
    correct_sentiment = [0]*len(sentiment)
    for _,row in validation_df.iterrows():
        correct_sentiment[phraseId.index(row['PhraseId'])] = row['Sentiment']
    return correct_sentiment

def evaluate(phraseId, sentiment, correct_sentiment):
    #Create confusion matrix
    confusion_matrix = np.zeros((5,5))
    
    for id in range(len(phraseId)):
        confusion_matrix[correct_sentiment[id]][sentiment[id]] += 1
    print(confusion_matrix)
    correct=0
    count=0
    for i in range(len(sentiment)):
        if sentiment[i]==correct_sentiment[i]:
            correct+=1
        count+=1
    print(correct,'/',count,' = ',correct/count)

def save_to_csv(test_phraseId, test_sentiment):
    filename = "Submission.csv"
    # f = open("nboutput.txt","w+")
    dict_to_convert = dict()
    dict_to_convert['PhraseId'] = test_phraseId
    dict_to_convert['Sentiment'] = test_sentiment
    df = pd.DataFrame(data = dict_to_convert)
    print("Saving")
    df.to_csv(path_or_buf=filename, header=['PhraseId', 'Sentiment'], index=False)




train_file = "sentiment-analysis-on-movie-reviews/train.tsv"
test_file = "sentiment-analysis-on-movie-reviews/test.tsv"
train_df, validation_df, test_df = get_data(train_file,test_file,0.1)
print(train_df.head())
senti_label, V, V_senti_freq, logprior_senti, loglikelihood_w_given = learn(train_df)
print(validation_df.head())
phraseId, sentiment = classify(validation_df, senti_label, V, V_senti_freq, logprior_senti, loglikelihood_w_given)
correct_sentiment = verify(phraseId, sentiment,validation_df)
evaluate(phraseId, sentiment, correct_sentiment)

test_phraseId, test_sentiment = classify(test_df, senti_label, V, V_senti_freq, logprior_senti, loglikelihood_w_given)
save_to_csv(test_phraseId, test_sentiment)

