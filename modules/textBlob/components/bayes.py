import pandas
import nltk
from functions.generateRandomSample import generate
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def train():
    # Download stopwords
    nltk.download('stopwords')

    # Read in the data into a pandas dataframe
    print("Reading data:")
    df = pandas.read_csv('../data/cleanDataset.csv')

    print("Starting data cleaning:")

    # Convert the 'review_text' column to string data type
    df['review_text'] = df['review_text'].astype(str)

    # Remove 'Early Access Review' from dataset
    df = df[~df['review_text'].str.contains('Early Access Review')]

    # Generate 5000 negative values
    negative_set = generate(df, "review_score", "neg", 5000)

    # Generate 5000 positive values
    positive_set = generate(df, "review_score", "pos", 5000)

    # Merge both sets
    training_set = pandas.concat([positive_set, negative_set], ignore_index=True)

    # Shuffle the dataframe
    training_set = training_set.sample(frac=1.0, random_state=42)

    # Make everything lowercase
    training_set['review_text'] = training_set['review_text'].str.lower()

    # Replace all ',' with ''
    training_set["review_text"]=training_set["review_text"].str.replace(',','', regex=True)

    # Replace all '.' with ''
    training_set["review_text"]=training_set["review_text"].str.replace('.','', regex=True)

    # Set stopwords for English language
    stop_words = set(stopwords.words('english'))

    # Define function to remove stopwords from text
    def remove_stopwords(text):
        words = text.split()
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        return ' '.join(filtered_words)

    # Apply the remove_stopwords function to the 'text' column in the DataFrame
    training_set['review_text'] = training_set['review_text'].apply(remove_stopwords)

    # Remove urls
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    training_set['review_text'] = training_set['review_text'].str.replace(urlPattern, '', regex=True)

    # Split into train and test split
    X_train, X_test, y_train, y_test = train_test_split(training_set['review_text'], training_set['review_score'], test_size=0.2, random_state=42)

    # Vectorise
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Get the feature importances from the trained classifier
    feature_importances = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    # Create a dictionary of features and their importances
    features = dict(zip(feature_names, feature_importances))

    # Sort the dictionary by importance (descending order)
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 pos features
    print("Top 5 positive features")
    for feature, importance in sorted_features[:5]:
        print("{} - {}".format(feature, importance))

    # Print the top 10 neg features
    print("Top 5 negative features")
    for feature, importance in sorted_features[-5:]:
        print("{} - {}".format(feature, importance))


    print()

    ## Convert the preprocessed text into numerical features
    text = "I really liked this"
    text_features = vectorizer.transform([text])

    # Predict the sentiment of the input text
    sentiment = clf.predict(text_features)[0]

    print(clf.predict(text_features)[0])
    print("Accuracy score: ", accuracy)

    return clf

def classify(clf, text):
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.transform([text])
    print(clf.predict(text_features)[0])



clf = train()
classify(clf, "I really like cheese")