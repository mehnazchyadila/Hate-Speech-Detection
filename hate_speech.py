from ProcessPre import process
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def read_files():
    train_data = []
    dev_data = []
    test_data = []

    # reads the training data
    with open('train.txt', 'r', encoding='utf-8') as train_file:
        for line in train_file:
            train_data.append(line)

    # reads the development data data
    with open('dev.txt', 'r', encoding='utf-8') as dev_file:
        for line in dev_file:
            dev_data.append(line)

    # reads the test data
    with open('test.txt', 'r', encoding='utf-8') as test_file:
        for line in test_file:
            test_data.append(line)

    return train_data, dev_data, test_data


def separate_labels(data):
    documents = []
    labels = []

    for line in data:
        splitted_line = line.split('\t', 2)
        # separate the labels and examples (docs) in different list
        labels.append(splitted_line[2])
        documents.append(splitted_line[1])
    return documents, labels


def separate_labels_test(data):
    ids = []
    documents = []

    for line in data:
        splitted_line = line.split('\t', 2)
        # separate the labels and examples (docs) in different list
        documents.append(splitted_line[1])
        ids.append(splitted_line[0])

    return documents, ids


def identity(X):
    return X


def vectorization_word(is_tfidf=True):
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if is_tfidf:
        vec = TfidfVectorizer(preprocessor=identity, lowercase=True,
                              tokenizer=identity, ngram_range=(2, 5), analyzer='word')
    else:
        vec = CountVectorizer(preprocessor=identity, lowercase=True,
                              tokenizer=identity, ngram_range=(2, 5), analyzer='word')

    return vec


def Naive_Bayes(train_docs, train_lbls, dev_docs, dev_lbls, test_docs, test_ids):
    vec = vectorization_word(is_tfidf=False)
    classifier = Pipeline([('vec', vec),
                           ('cls', MultinomialNB())])
    classifier.fit(train_docs, train_lbls)
    classifier.fit(dev_docs, dev_lbls)
    predict_dev = classifier.predict(dev_docs)
    predict_test = classifier.predict(test_docs)
    with open("predict_test_NB.txt", mode="w+", encoding="utf-8") as f:
        f.write("doc_id" + "\t" + "predicted_label" + "\n")
        for test_ids, predict_test in zip(test_ids, predict_test):
            f.write(test_ids + "\t" + predict_test)
    print("FOR DEVLOPMENT DATA:")
    evaluation_results(dev_lbls, predict_dev, classifier)


def SVM_static(train_docs, train_lbls, dev_docs, dev_lbls, test_docs, test_ids):
    # calls the vectorization function
    vec = vectorization_word(is_tfidf=True)
    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', SVC(kernel='rbf'))])
    classifier.fit(train_docs, train_lbls)
    classifier.fit(dev_docs, dev_lbls)
    predict = classifier.predict(dev_docs)
    predict_test = classifier.predict(test_docs)
    with open("predict_test_SVM.txt", mode="w+", encoding="utf-8") as f:
        f.write("doc_id" + "\t" + "predicted_label" + "\n")
        for test_ids, predict_test in zip(test_ids, predict_test):
            f.write(test_ids + "\t" + predict_test)

    evaluation_results(dev_lbls, predict, classifier)




def evaluation_results(dev_lbls, predict, classifier):
    # Compare the accuracy of the output (predict) with the class labels of the original test set (test_lbls)
    print("Accuracy = ", accuracy_score(dev_lbls, predict))

    # Report on the precision, recall, f1-score of the output (Yguess) with the class labels of the original test set (Ytest)
    print(classification_report(dev_lbls, predict, labels=classifier.classes_, target_names=None, sample_weight=None,
                                digits=3))

    Confusion_Matrix = confusion_matrix(
        dev_lbls, predict, labels=classifier.classes_)
    print("Confusion Matrix :\n", Confusion_Matrix)


def main():
    print('......READING THE DATASET....')
    train_data, dev_data, test_data = read_files()

    train_docs, train_lbls = separate_labels(train_data)
    dev_docs, dev_lbls = separate_labels(dev_data)
    test_docs, test_ids = separate_labels_test(test_data)

    print('\n\n......PREPROCESSING....')
    # only tokenizing the documents
    preprocessed_train_docs = process(train_docs)
    # print(preprocessed_train_docs)

    preprocessed_dev_docs = process(dev_docs)
    # print(preprocessed_dev_docs)

    preprocessed_test_docs = process(test_docs)
    # print(preprocessed_dev_docs)

    print('\n.....TRAINING THE NB CLASSIFIER....\n\n')
    Naive_Bayes(train_docs=preprocessed_train_docs, train_lbls=train_lbls,
                dev_docs=preprocessed_dev_docs, dev_lbls=dev_lbls,
                test_docs=preprocessed_test_docs, test_ids=test_ids)

    with open("predict_test_NB.txt", mode="r", encoding="utf-8") as f:
        data = f.read()
        hate = data.count("HATE")
        offn = data.count("OFFN")
        prfn = data.count("PRFN")
        none = data.count("NONE")
    print("Word count for test data label with NB using Count Vectorizer")
    print("___________________________________________")
    print("|  HATE  | OFFENSIVE |  PROFANE  |  NONE  |")
    print("|  ", hate, " |   ", offn, "   |  ", prfn, "  |  ", none, "  |")
    print("___________________________________________")

    print('\nTraining The SVM Classifier....\n\n')
    SVM_static(train_docs=preprocessed_train_docs, train_lbls=train_lbls,
               dev_docs=preprocessed_dev_docs, dev_lbls=dev_lbls,
               test_docs=preprocessed_test_docs, test_ids=test_ids)

    with open("predict_test_SVM.txt", mode="r", encoding="utf-8") as f:
        data = f.read()
        hate = data.count("HATE")
        offn = data.count("OFFN")
        prfn = data.count("PRFN")
        none = data.count("NONE")
    print("Word count for test data label with SVM using TFIDF Vectorizer")
    print("___________________________________________")
    print("|  HATE  | OFFENSIVE |  PROFANE  |  NONE  |")
    print("|  ", hate, " |   ", offn, "   |  ", prfn, "  |  ", none, "  |")
    print("___________________________________________")




if __name__ == "__main__":
    main()