import nltk
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

class Editor:

    def __init__(self):
        word_list = []

        with open('../sumarizarea-documentelor/stop_words', 'r') as f:
            words = f.readlines()

        for word in words:
            edited = word[:-1]
            word_list.append(edited)

        self.__stop_words = word_list


    def sentence_tokenize(self, text):
        sentences = sent_tokenize(text)
        return sentences

    def word_tokenize(self, text):

        tokenizer = nltk.RegexpTokenizer(r"\w+")

        words = tokenizer.tokenize(text)

        return words

    def remove_stop_words(self, text):

        result = []

        for word in text:
            if word in self.__stop_words:
                continue
            result.append(word)

        return result

    def lemmatize(self, text):

        result = []
        lemmatizer = WordNetLemmatizer()

        for word in text:
            lemmatized = lemmatizer.lemmatize(word)
            result.append(lemmatized)

        return result
