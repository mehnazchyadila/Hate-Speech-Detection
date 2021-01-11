import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

def text_lowercase(text):
    return text.lower()


# Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

def lemmatization(doc):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(doc)
    return lemma
# convert number into words
def convert_number(text):
    p = inflect.engine()
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []

    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)

            # append the word as it is
        else:
            new_string.append(word)

            # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str

def remove_special_char(text):
    c = re.sub('[!,*)@#(&$_?]', '', text)
    return c
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text



# remove punctuation
def remove_punctuation(text):
    # map punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation), '')
    return text.translate(translator)

# remove whitespace from text
def remove_whitespace(text):
    return " ".join(text.split())


def tokenie_sentence(text):
    word_tokens = word_tokenize(text)
    return word_tokens


# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = tokenie_sentence(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


# we are using NLTK stemmer to stem multiple words into root
def apply_stemmer(doc):
    stemmer = PorterStemmer()

    roots = [stemmer.stem(plural) for plural in doc]

    return roots


def process(documents):
    # combined_doc = []
    lowercased_doc = []
    for sent in documents:
        lowercased_sen = text_lowercase(sent)
        lowercased_doc.append(lowercased_sen)

    # print("Output of Lowercasing Operation:\n",temp_doc)

    temp_doc = []
    for sent in documents:
        temp_doc.append(remove_numbers(sent))

    # print("Output of Removed Number Operation:\n", temp_doc)

    temp_doc = []
    for sent in documents:
        temp_doc.append(convert_number(sent))

    # print("Output of Convert Number Operation:\n", temp_doc)

    temp_doc = []
    for sent in documents:
        temp_doc.append(lemmatization(sent))

    # print("Output of Lemmatization Operation:\n", temp_doc)

    temp_doc = []
    for sent in documents:
        temp_doc.append(remove_punctuation(sent))

    # print("Output of Remove Punctuations Operation:\n", temp_doc)

    # documents = temp_doc
    temp_doc = []
    for sent in documents:
        temp_doc.append(remove_whitespace(sent))

    # print("Output of Removing Multiple Whitespaces Operation:\n", temp_doc)

    # documents = lowercased_doc
    tokenized_doc = []
    for sent in documents:
        tokenized_doc.append(tokenie_sentence(sent))

    # return tokenized_doc

    # documents = lowercased_doc
    stopwords_removed_doc = []
    for sent in documents:
        stopwords_removed_doc.append(remove_stopwords(sent))

    # print("Output of Tokenization Operation:\n", temp_doc)

    # return stopwords_removed_doc

    return documents
