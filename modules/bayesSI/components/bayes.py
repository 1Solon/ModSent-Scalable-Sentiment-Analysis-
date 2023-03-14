import nltk.classify.util
from joblib import dump, load
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Splits a sentence into a dict of words and removes stopwords
def generate_word_features(words):
    stopwords.words('english')[:16]
    useful_words = [
        word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

# Trains the classifier
def train():
    # Downloads the "movie_reviews" standard dataset
    nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews

    # Generates a set of negative reviews
    print("Generate negative review set:")
    neg_reviews = [] 
    for fileid in movie_reviews.fileids('neg'):
        words = movie_reviews.words(fileid)
        neg_reviews.append((generate_word_features(words), "negative"))

    # Generates a set of positive reviews
    print("Generate positive review set:")
    pos_reviews = []
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append((generate_word_features(words), "positive"))

    # Splits into train and test set
    # 3/4 as train, 1/4 as test
    print("Generate text/train split:")
    train_set = neg_reviews[:750] + pos_reviews[:750]
    test_set = neg_reviews[750:] + pos_reviews[750:]
    print(len(train_set),  len(test_set))

    # create the NaiveBayesClassifier
    print("Generate NaiveBayesClassifier")
    classifier = NaiveBayesClassifier.train(train_set)

    # Dump classified files to the data directory
    print("Dump classifier to data directory:")
    dump(classifier, '../data/bayes.joblib')
    dump(test_set, '../data/test_set.joblib')

# Returns the accuracy of the dataset
def accuracy():
    # Checks if joblibs exist, if not, train classifier
    try:
        classifier = load('../data/bayes.joblib')
        test_set = load('../data/test_set.joblib')
    except Exception as e:
        print(e)
        train()
        accuracy()

    # Print accuracy
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    
    return accuracy * 100


# Classifies a given piece of text
def classify(text: str):
    # Checks if joblibs exist, if not, train classifier
    try:
        classifier = load('../data/bayes.joblib')
    except:
        print("Classifier file does not exist, begin training of classifier:")
        train()
        classify(text)

    # Tonkenise input text then return 'pos' or 'neg'
    text = word_tokenize(text)
    text = generate_word_features(text)
    return classifier.classify(text)