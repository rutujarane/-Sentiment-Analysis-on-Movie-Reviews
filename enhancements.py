import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class Enhanced:
    #Removing stopwords:
    def remove_stopwords(self,token_list):
        stopwords_list = {'before', 'o', "it's", 'our', 'against', 'not', 'you', 'are', 'over', 'herself', 'y', "haven't", 'had', 'all', 'again', 'for', 'being', 'why', 'no', 'ma', 'or', 'from', 'me', 'above', "you're", 'through', "aren't", "mightn't", 'themselves', 'further', "shouldn't", 'wouldn', 'them', 'should', 'then', 'other', 'yourself', 'than', "won't", "wasn't", 'haven', 'won', 'hasn', 'nor', 'ain', 'himself', 'i', 't', "weren't", 'until', "needn't", 'some', 'to', 's', "should've", 'as', "doesn't", 'between', 'such', "she's", "you'll", 'has', 'because', 'most', 'ourselves', 'mightn', 'now', 'out', 'can', 'so', "hadn't", 'of', 'on', 'have', 'be', 'doing', 'any', 'with', 'here', 'and', 'isn', 'doesn', "you'd", 'she', 'into', 'didn', 'yours', 'during', 'my', 'yourselves', 'while', 'aren', 'ours', 'him', 'this', 'having', 'shouldn', 'is', 'both', "shan't", "that'll", 'these', 'up', 'd', 'few', 'needn', 'll', 're', 'but', 'if', 'when', 'who', 'off', 'more', 'we', "wouldn't", 'they', 'weren', 'their', 'been', 'very', 'was', 'its', "you've", 'were', 'hadn', 've', 'her', 'those', 'how', 'by', "isn't", 'whom', 'he', 'which', 'hers', 'does', 'below', 'did', 'm', 'an', 'your', 'only', 'the', "mustn't", 'same', 'about', 'what', 'down', 'itself', 'am', 'don', 'wasn', "didn't", 'that', 'each', 'after', 'shan', 'there', 'do', 'too', 'just', 'will', 'it', 'theirs', 'his', "don't", 'couldn', 'under', "hasn't", 'where', 'mustn', 'at', 'in', "couldn't", 'myself', 'once', 'a', 'own'}
        removed_list = []
        for word in token_list:
            if word not in stopwords_list:
                removed_list.append(word)
        token_list = []
        token_list = removed_list
        return token_list

    #Converting to lowercase:
    def lower_case(self,token_list):
        token_list = [word.lower() for word in token_list]
        return token_list

    #Removing punctuation:
    def remove_punctuation(self,token_list):
        token_list = [word for word in token_list if word.isalnum]
        return token_list

    #Stemming:
    def stemming(self,token_list):
        p = PorterStemmer()
        stem_list = [p.stem(word) for word in token_list]
        token_list = []
        token_list = stem_list
        return token_list

    #Lemmatize:
    def lemmatize(self,token_list):
        l = WordNetLemmatizer()
        token_list = [l.lemmatize(word) for word in token_list]
        return token_list

    #Add-k smoothing:
    def add_k_smoothing(self,add_what):
        if add_what == "add_k":
            # return 0.00005
            return 0.005
        return 1

    #Good Turing Frequency of Frequencies Generation:
    def good_turing_fof(self, V):
        N = dict()
        total = 0
        for word in V:
            if V[word] in N:
                N[V[word]] += 1
            else:
                N[V[word]] = 1
            total += 1
        return N, total

    #Good Turing Answers:
    def good_turing_values(self,sum_ham,sum_spam,w,V,N,total_freq,lh,ls):
        if V[w] > 10:
            sum_ham = sum_ham + lh
            sum_spam = sum_spam + ls
        else:
            if (V[w]+1) in N:
                sum_ham = sum_ham + (((V[w]+1) * (N[V[w]+1]/ N[V[w]])) / total_freq)
                sum_spam = sum_spam + (((V[w]+1) * (N[V[w]+1] / N[V[w]])) / total_freq)
            else:
                sum_ham = sum_ham + lh
                sum_spam = sum_spam + ls
        return sum_ham, sum_spam
