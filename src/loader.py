import os
import csv
from document import Document

NEWS_CLASSES = ['business', 'entertainment', 'politics', 'sport', 'tech']


class Loader:

    def __init__(self):
        self.__business_docs = []
        self.__entertainment_docs = []
        self.__politics_docs = []
        self.__sport_docs = []
        self.__tech_docs = []

    def get_content(self):
        return self.__business_docs, self.__entertainment_docs, \
            self.__politics_docs, self.__sport_docs, self.__tech_docs

    def add_to_business(self, doc):
        self.__business_docs.append(doc)

    def add_to_entertainment(self, doc):
        self.__entertainment_docs.append(doc)

    def add_to_politics(self, doc):
        self.__politics_docs.append(doc)

    def add_to_sport(self, doc):
        self.__sport_docs.append(doc)

    def add_to_tech(self, doc):
        self.__tech_docs.append(doc)

    def show_contents(self):
        for bdoc in self.__business_docs:
            bdoc.show_info()
        for edoc in self.__entertainment_docs:
            edoc.show_info()
        for pdoc in self.__politics_docs:
            pdoc.show_info()
        for sdoc in self.__sport_docs:
            sdoc.show_info()
        for tdoc in self.__tech_docs:
            tdoc.show_info()

    def to_csv(self):

        with open('data.csv', mode='w') as data_file:
            writer = csv.writer(data_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for b_doc in self.__business_docs:
                content = b_doc.get_content()
                summary = b_doc.get_summary()
                dtype = b_doc.get_dtype()
                writer.writerow([content, summary, dtype])
            for e_doc in self.__entertainment_docs:
                content = e_doc.get_content()
                summary = e_doc.get_summary()
                dtype = e_doc.get_dtype()
                writer.writerow([content, summary, dtype])
            for p_doc in self.__politics_docs:
                content = p_doc.get_content()
                summary = p_doc.get_summary()
                dtype = p_doc.get_dtype()
                writer.writerow([content, summary, dtype])
            for s_doc in self.__sport_docs:
                content = s_doc.get_content()
                summary = s_doc.get_summary()
                dtype = s_doc.get_dtype()
                writer.writerow([content, summary, dtype])
            for t_doc in self.__tech_docs:
                content = t_doc.get_content()
                summary = t_doc.get_summary()
                dtype = t_doc.get_dtype()
                writer.writerow([content, summary, dtype])

    def load(self, path, doc_type):
        path_to_articles = path + "News Articles/" + doc_type
        path_to_summaries = path + "Summaries/" + doc_type

        articles = os.listdir(path_to_articles)
        summaries = os.listdir(path_to_summaries)

        for i in range(0, len(articles)):

            f_article = open(path_to_articles + "/" + articles[i])
            f_summary = open(path_to_summaries + "/" + summaries[i])

            art = f_article.read()
            summ = f_summary.read()

            doc = Document(art, summ, doc_type)

            exec('self.add_to_' + doc_type + '(doc)')

    def load_all(self, path):
        for news_class in NEWS_CLASSES:
            self.load(path, news_class)
