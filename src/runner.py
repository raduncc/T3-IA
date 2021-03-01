import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from loader import Loader
from editor import Editor
from copy import deepcopy
import nltk
from rouge_score import rouge_scorer
from sklearn.model_selection import KFold
import numpy as np
from math import sqrt

CLASSES = ['business', 'entertainment', 'politics', 'sport', 'tech']


def execute_classify(do_k_fold):

    ldr = Loader()

    column_names = ["content", "summary", "type"]
    data = pd.read_csv('data.csv', names=column_names)

    X = data["content"]
    y = data["type"]

    if do_k_fold:

        precs1 = []
        recs1 = []

        precs2 = []
        recs2 = []

        precs3 = []
        recs3 = []

        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), False, False)
            precs1.append(prec)
            recs1.append(rec)
            print('Classification - stop words not eliminated, no lemmatize\nConfusion matrix is:\n' +
                  str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
            print()
            (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), True, False)
            precs2.append(prec)
            recs2.append(rec)
            print('Classification - stop words eliminated, no lemmatize\nConfusion matrix is:\n' +
                  str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
            print()
            (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test),
                                       deepcopy(y_train), deepcopy(y_test), True, True)
            precs3.append(prec)
            recs3.append(rec)
            print('Classification - stop words eliminated, lemmatize\nConfusion matrix is:\n' +
                  str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
            print()

            print('============')

        mean_p1 = sum(precs1) / 5
        mean_p2 = sum(precs2) / 5
        mean_p3 = sum(precs3) / 5

        mean_r1 = sum(recs1) / 5
        mean_r2 = sum(recs2) / 5
        mean_r3 = sum(recs3) / 5

        s = 0

        for e in precs1:
            s += (e - mean_p1) ** 2

        stdev_p1 = sqrt(s / 5)

        s = 0

        for e in precs2:
            s += (e - mean_p2) ** 2

        stdev_p2 = sqrt(s / 5)

        s = 0

        for e in precs3:
            s += (e - mean_p3) ** 2

        stdev_p3 = sqrt(s/5)

        s = 0

        for e in recs1:
            s += (e - mean_r1) ** 2

        stdev_r1 = sqrt(s/5)

        s = 0

        for e in recs2:
            s += (e - mean_r2) ** 2

        stdev_r2 = sqrt(s/5)

        s = 0

        for e in recs3:
            s += (e - mean_r3) ** 2

        stdev_r3 = sqrt(s/5)

        print('CLASSIFICATION, NO STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p1) +
              '\n-std_dev: ' + str(stdev_p1) + '\nRECALL:\n-mean: ' + str(mean_r1) + '\n-std_dev: ' + str(stdev_r1))
        print('CLASSIFICATION, STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p2) +
              '\n-std_dev: ' + str(stdev_p2) + '\nRECALL:\n-mean: ' + str(mean_r2) + '\n-std_dev: ' + str(stdev_r2))
        print('CLASSIFICATION, STOP WORDS ELIMINATED, LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p3) +
              '\n-std_dev: ' + str(stdev_p3) + '\nRECALL:\n-mean: ' + str(mean_r3) + '\n-std_dev: ' + str(stdev_r3))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)

        (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), False, False)
        print('Classification - stop words not eliminated, no lemmatize\nConfusion matrix is:\n' +
              str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
        print()
        (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), True, False)
        print('Classification - stop words eliminated, no lemmatize\nConfusion matrix is:\n' +
              str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
        print()
        (cm, prec, rec) = classify(deepcopy(X_train), deepcopy(X_test),
                                   deepcopy(y_train), deepcopy(y_test), True, True)
        print('Classification - stop words eliminated, lemmatize\nConfusion matrix is:\n' +
              str(cm) + '\nPrecision is: ' + str(prec) + '\nRecall is: ' + str(rec))
        print()

        print('============')


def execute_summarise(do_k_fold):

    ldr = Loader()

    column_names = ["content", "summary", "type"]
    data = pd.read_csv('data.csv', names=column_names)

    X = data["content"]
    y = data["summary"]

    if do_k_fold:

        precs1 = []
        recs1 = []

        precs2 = []
        recs2 = []

        precs3 = []
        recs3 = []

        precs4 = []
        recs4 = []

        precs5 = []
        recs5 = []

        precs6 = []
        recs6 = []

        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print('===MONOGRAMS===')

            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), False, False, False)
            precs1.append(prec)
            recs1.append(rec)
            print('Summarisation - stop words not eliminated, no lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()
            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), True, False, False)
            precs2.append(prec)
            recs2.append(rec)
            print('Summarisation - stop words eliminated, no lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()
            precs3.append(prec)
            recs3.append(rec)
            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test),
                                    deepcopy(y_train), deepcopy(y_test), True, True, False)
            print('Summarisation - stop words eliminated, lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()

            print('===BIGRAMS===')

            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), False, False, True)
            precs4.append(prec)
            recs4.append(rec)
            print('Summarisation - stop words not eliminated, no lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()
            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
                y_train), deepcopy(y_test), True, False, True)
            precs5.append(prec)
            recs5.append(rec)
            print('Summarisation - stop words eliminated, no lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()
            (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test),
                                    deepcopy(y_train), deepcopy(y_test), True, True, True)
            precs6.append(prec)
            recs6.append(rec)
            print('Summarisation - stop words eliminated, lemmatize\nPrecision is: ' +
                  str(prec) + '\nRecall is: ' + str(rec))
            print()

        mean_p1 = sum(precs1)/5
        mean_p2 = sum(precs2)/5
        mean_p3 = sum(precs3)/5
        mean_p4 = sum(precs4)/5
        mean_p5 = sum(precs5)/5
        mean_p6 = sum(precs6)/5

        mean_r1 = sum(recs1)/5
        mean_r2 = sum(recs2)/5
        mean_r3 = sum(recs3)/5
        mean_r4 = sum(recs4)/5
        mean_r5 = sum(recs5)/5
        mean_r6 = sum(recs6)/5

        s1p = 0
        s2p = 0
        s3p = 0
        s4p = 0
        s5p = 0
        s6p = 0

        s1r = 0
        s2r = 0
        s3r = 0
        s4r = 0
        s5r = 0
        s6r = 0

        for i in range(len(precs1)):
            s1p += (mean_p1 - precs1[i]) ** 2
            s2p += (mean_p2 - precs2[i]) ** 2
            s3p += (mean_p3 - precs3[i]) ** 2
            s4p += (mean_p4 - precs4[i]) ** 2
            s5p += (mean_p5 - precs5[i]) ** 2
            s6p += (mean_p6 - precs6[i]) ** 2

            s1r += (mean_r1 - recs1[i]) ** 2
            s2r += (mean_r2 - recs2[i]) ** 2
            s3r += (mean_r3 - recs3[i]) ** 2
            s4r += (mean_r4 - recs4[i]) ** 2
            s5r += (mean_r5 - recs5[i]) ** 2
            s6r += (mean_r6 - recs6[i]) ** 2

        stdev_p1 = sqrt(s1p/5)
        stdev_p2 = sqrt(s2p/5)
        stdev_p3 = sqrt(s3p/5)
        stdev_p4 = sqrt(s4p/5)
        stdev_p5 = sqrt(s5p/5)
        stdev_p6 = sqrt(s6p/5)

        stdev_r1 = sqrt(s1r/5)
        stdev_r2 = sqrt(s2r/5)
        stdev_r3 = sqrt(s3r/5)
        stdev_r4 = sqrt(s4r/5)
        stdev_r5 = sqrt(s5r/5)
        stdev_r6 = sqrt(s6r/5)

        print('MONOGRAMS')
        print('SUMMARISATION, NO STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p1) +
                '\n-std_dev: ' + str(stdev_p1) + '\nRECALL:\n-mean: ' + str(mean_r1) + '\n-std_dev: ' + str(stdev_r1))
        print('SUMMARISATION, STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p2) +
                '\n-std_dev: ' + str(stdev_p2) + '\nRECALL:\n-mean: ' + str(mean_r2) + '\n-std_dev: ' + str(stdev_r2))
        print('SUMMARISATION, STOP WORDS ELIMINATED, LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p3) +
                '\n-std_dev: ' + str(stdev_p3) + '\nRECALL:\n-mean: ' + str(mean_r3) + '\n-std_dev: ' + str(stdev_r3))
        print('BIGRAMS')
        print('SUMMARISATION, NO STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p4) +
                '\n-std_dev: ' + str(stdev_p4) + '\nRECALL:\n-mean: ' + str(mean_r4) + '\n-std_dev: ' + str(stdev_r4))
        print('SUMMARISATION, STOP WORDS ELIMINATED, NO LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p5) +
                '\n-std_dev: ' + str(stdev_p5) + '\nRECALL:\n-mean: ' + str(mean_r5) + '\n-std_dev: ' + str(stdev_r5))
        print('SUMMARISATION, STOP WORDS ELIMINATED, LEMMATIZE:\nPRECISION:\n-mean: ' + str(mean_p6) +
                '\n-std_dev: ' + str(stdev_p6) + '\nRECALL:\n-mean: ' + str(mean_r6) + '\n-std_dev: ' + str(stdev_r6))
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25)

        print('===MONOGRAMS===')

        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), False, False, False)
        print('Summarisation - stop words not eliminated, no lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()
        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), True, False, False)
        print('Summarisation - stop words eliminated, no lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()
        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test),
                                deepcopy(y_train), deepcopy(y_test), True, True, False)
        print('Summarisation - stop words eliminated, lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()

        print('===BIGRAMS===')

        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), False, False, True)
        print('Summarisation - stop words not eliminated, no lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()
        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test), deepcopy(
            y_train), deepcopy(y_test), True, False, True)
        print('Summarisation - stop words eliminated, no lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()
        (prec, rec) = summarize(deepcopy(X_train), deepcopy(X_test),
                                deepcopy(y_train), deepcopy(y_test), True, True, True)
        print('Summarisation - stop words eliminated, lemmatize\nPrecision is: ' +
              str(prec) + '\nRecall is: ' + str(rec))
        print()

        print('============')


def classify(X_train, X_test, y_train, y_test, skip_stop_words, do_lematize):

    editor = Editor()

    d_words = {'business': {}, 'entertainment': {},
               'politics': {}, 'sport': {}, 'tech': {}}

    d_articles = {'business': 0, 'entertainment': 0,
                  'politics': 0, 'sport': 0, 'tech': 0}

    total_words = {'business': 0, 'entertainment': 0,
                   'politics': 0, 'sport': 0, 'tech': 0}

    vocabulary = set()

    for i in range(len(X_train)):

        content = X_train.iloc[i]
        dtype = y_train.iloc[i]

        edited_content = editor.word_tokenize(content)

        if skip_stop_words:
            edited_content = editor.remove_stop_words(edited_content)

        if do_lematize:
            edited_content = editor.lemmatize(edited_content)

        d_articles[dtype] = d_articles[dtype] + 1

        for word in edited_content:

            total_words[dtype] = total_words[dtype] + 1

            vocabulary.add(word)

            if word in d_words[dtype]:
                d_words[dtype][word] = d_words[dtype][word] + 1
            else:
                d_words[dtype][word] = 1

    alpha = 1
    voc_len = len(vocabulary)

    total_articles = len(X_train)

    total_words_all = total_words['business'] + total_words['entertainment'] + \
        total_words['tech'] + total_words['politics'] + total_words['sport']

    line_class_dict = {'business': 0, 'entertainment': 1,
                       'politics': 2, 'sport': 3, 'tech': 4}

    y_true = []
    y_pred = []

    for i in range(len(X_test)):

        content = X_test.iloc[i]
        actual_dtype = y_test.iloc[i]

        y_true.append(line_class_dict[actual_dtype])

        edited_content = editor.word_tokenize(content)

        if skip_stop_words:
            edited_content = editor.remove_stop_words(edited_content)

        if do_lematize:
            edited_content = editor.lemmatize(edited_content)

        maximum_c = float('-inf')
        final_class = None

        for c in CLASSES:

            c_map = math.log(d_articles[c] / total_articles)

            for word in edited_content:

                if word in d_words[c]:
                    no_aps = d_words[c][word]
                else:
                    no_aps = 0

                c_map += math.log((no_aps + alpha) /
                                  (total_words[c] + alpha + voc_len))

            if c_map > maximum_c:
                maximum_c = c_map
                final_class = c

        y_pred.append(line_class_dict[final_class])

    return metrics.confusion_matrix(y_true, y_pred), \
        metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)['weighted avg']['precision'], \
        metrics.classification_report(y_true, y_pred, digits=3, output_dict=True)[
        'weighted avg']['recall']


def process_content():

    res_b = []
    res_e = []
    res_p = []
    res_s = []
    res_t = []

    b_file = open('business', 'wb')
    e_file = open('entertainment', 'wb')
    p_file = open('politics', 'wb')
    s_file = open('sport', 'wb')
    t_file = open('tech', 'wb')

    loader = Loader()
    editor = Editor()
    path = '../sumarizarea-documentelor/BBC News Summary/'

    loader.load_all(path)

    (b_docs, e_docs, p_docs, s_docs, t_docs) = loader.get_content()

    for b_doc in b_docs:
        result = editor.word_tokenize(b_doc.get_content())
        result = editor.remove_stop_words(result)
        result = editor.lemmatize(result)
        res_b.append(result)

    pickle.dump(res_b, b_file)

    for e_doc in e_docs:
        result = editor.word_tokenize(e_doc.get_content())
        result = editor.remove_stop_words(result)
        result = editor.lemmatize(result)
        res_e.append(result)

    pickle.dump(res_e, e_file)

    for p_doc in p_docs:
        result = editor.word_tokenize(p_doc.get_content())
        result = editor.remove_stop_words(result)
        result = editor.lemmatize(result)
        res_p.append(result)

    pickle.dump(res_p, p_file)

    for s_doc in s_docs:
        result = editor.word_tokenize(s_doc.get_content())
        result = editor.remove_stop_words(result)
        result = editor.lemmatize(result)
        res_s.append(result)

    pickle.dump(res_s, s_file)

    for t_doc in t_docs:
        result = editor.word_tokenize(t_doc.get_content())
        result = editor.remove_stop_words(result)
        result = editor.lemmatize(result)
        res_t.append(result)

    pickle.dump(res_t, t_file)

    b_file.close()
    e_file.close()
    p_file.close()
    s_file.close()
    t_file.close()


def get_from_pickle():
    b_file = open('business', 'rb')
    e_file = open('entertainment', 'rb')
    p_file = open('politics', 'rb')
    s_file = open('sport', 'rb')
    t_file = open('tech', 'rb')

    b_list = pickle.load(b_file)
    e_list = pickle.load(e_file)
    p_list = pickle.load(p_file)
    s_list = pickle.load(s_file)
    t_list = pickle.load(t_file)

    b_file.close()
    e_file.close()
    p_file.close()
    s_file.close()
    t_file.close()

    return b_list, e_list, p_list, s_list, t_list


def summarize(X_train, X_test, y_train, y_test, skip_stop_words, do_lematize, bigrams):

    editor = Editor()

    classes_sentences = {'in-summary': 0, 'out-of-summary': 0}

    word_occurences = {'in-summary': {}, 'out-of-summary': {}}

    total_word_occurences = {}

    total_words = {'in-summary': 0, 'out-of-summary': 0}

    vocabulary = set()

    total_sentences = 0
    tw = 0

    for i in range(len(X_train)):

        content = X_train.iloc[i]
        summary = y_train.iloc[i]

        sentences_content = editor.sentence_tokenize(content)
        sentences_summary = editor.sentence_tokenize(summary)

        total_sentences += len(sentences_content)

        classes_sentences['in-summary'] += len(sentences_summary)
        classes_sentences['out-of-summary'] += len(
            sentences_content) - len(sentences_summary)

        for s in sentences_content:
            words = editor.word_tokenize(s)

            if skip_stop_words:
                words = editor.remove_stop_words(words)

            if do_lematize:
                words = editor.lemmatize(words)

            if bigrams:
                words = list(nltk.bigrams(words))

            tw += len(words)

            for word in words:

                vocabulary.add(word)

                if word in total_word_occurences:
                    total_word_occurences[word] += 1
                else:
                    total_word_occurences[word] = 1

        for s in sentences_summary:
            words = editor.word_tokenize(s)

            if skip_stop_words:
                words = editor.remove_stop_words(words)

            if do_lematize:
                words = editor.lemmatize(words)

            if bigrams:
                words = list(nltk.bigrams(words))

            for word in words:
                if word in word_occurences['in-summary']:
                    word_occurences['in-summary'][word] += 1
                else:
                    word_occurences['in-summary'][word] = 1

    for word in total_word_occurences:
        if word in word_occurences['in-summary']:
            word_occurences['out-of-summary'][word] = total_word_occurences[word] - \
                word_occurences['in-summary'][word]
        else:
            word_occurences['out-of-summary'][word] = total_word_occurences[word]

    for c in word_occurences:
        for w in word_occurences[c]:
            total_words[c] += word_occurences[c][w]

    y_true = []
    y_pred = []

    alpha = 1
    voc_len = len(vocabulary)

    for i in range(len(X_test)):

        content = X_test.iloc[i]
        actual_summary = y_test.iloc[i]

        y_true.append(actual_summary)
        sentences = editor.sentence_tokenize(content)

        sent_dict = {}

        computed_summary = ""

        average = 0

        score_dictionary = {}

        for sentence in sentences:

            words = editor.word_tokenize(sentence)

            if skip_stop_words:
                words = editor.remove_stop_words(words)

            if do_lematize:
                words = editor.lemmatize(words)

            if bigrams:
                words = list(nltk.bigrams(words))

            c_map = math.log(classes_sentences['in-summary'] / total_sentences)

            for word in words:

                if word in word_occurences['in-summary']:
                    no_aps = word_occurences['in-summary'][word]
                else:
                    no_aps = 0

                c_map += math.log((no_aps+alpha) /
                                  (total_words['in-summary'] + alpha + voc_len))

            score_dictionary[sentence] = c_map

            average += c_map

        average /= len(sentences)

        for sentence in score_dictionary:
            if score_dictionary[sentence] > average:
                computed_summary += sentence

        y_pred.append(computed_summary)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    precision_r1 = 0
    recall_r1 = 0

    precision_r2 = 0
    recall_r2 = 0

    for i in range(len(y_pred)):
        score = scorer.score(y_pred[i], y_true[i])

        precision_r1 += score['rouge1'].precision
        precision_r2 += score['rouge2'].precision

        recall_r1 += score['rouge1'].recall
        recall_r2 += score['rouge2'].recall

    precision_r1 /= len(y_pred)
    precision_r2 /= len(y_pred)

    recall_r1 /= len(y_pred)
    recall_r2 /= len(y_pred)

    if bigrams:
        return precision_r2, recall_r2
    else:
        return precision_r1, recall_r1


if __name__ == "__main__":
    
    # daca vrem sa cream un nou fisier csv cu datele
    # x = Loader()
    # x.load_all('../sumarizarea-documentelor/BBC News Summary/')
    # x.to_csv()


    # (a, b, c, d, e) = get_from_pickle()
    # print(a[1])
    
    # execute_classify(False)
    # print('===')
    # execute_classify(True)
    # print('===')
    # execute_summarise(False)
    # print('===')
    execute_summarise(True)
    # print('===')
