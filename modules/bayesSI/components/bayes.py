from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from math import *
import numpy
import zipfile
import string
import re
import pandas as pd
import nltk
import warnings
warnings.filterwarnings('ignore')

def train():
    # Unzip dataset for use
    with zipfile.ZipFile("../data/zippedData.zip","r") as zip_ref:
        zip_ref.extractall("../data")

    # Load dataset
    tweets = pd.read_csv('../data/zippedData/trainTweet.csv', encoding='latin',
                        names=['sentiment', 'id', 'date', 'query', 'user', 'tweet'])

    # Change format of positives and negatives
    tweets['sentiment'] = tweets['sentiment'].replace(4, 1)

    # Remove unused columns
    tweets.drop(['date', 'query', 'user'], axis=1, inplace=True)
    tweets.drop('id', axis=1, inplace=True)

    # Make sure text is in strings
    tweets['tweet'] = tweets['tweet'].astype('str')

    # Make sure stopwords and other packages are up to date and downloaded
    nltk.download('stopwords')
    nltk.download('wordnet')
    stopword = set(stopwords.words('english'))

    # Define urlPattern
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'

    # Define things to clean
    some = 'amp,today,tomorrow,going,girl'

    # Expand abreviations
    print("Cleaning dataset")
    def process_tweets(tweet):
    # Lower Casing
        tweet = re.sub(r"he's", "he is", tweet)
        tweet = re.sub(r"there's", "there is", tweet)
        tweet = re.sub(r"We're", "We are", tweet)
        tweet = re.sub(r"That's", "That is", tweet)
        tweet = re.sub(r"won't", "will not", tweet)
        tweet = re.sub(r"they're", "they are", tweet)
        tweet = re.sub(r"Can't", "Cannot", tweet)
        tweet = re.sub(r"wasn't", "was not", tweet)
        tweet = re.sub(r"don\x89Ûªt", "do not", tweet)
        tweet = re.sub(r"aren't", "are not", tweet)
        tweet = re.sub(r"isn't", "is not", tweet)
        tweet = re.sub(r"What's", "What is", tweet)
        tweet = re.sub(r"haven't", "have not", tweet)
        tweet = re.sub(r"hasn't", "has not", tweet)
        tweet = re.sub(r"There's", "There is", tweet)
        tweet = re.sub(r"He's", "He is", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"You're", "You are", tweet)
        tweet = re.sub(r"I'M", "I am", tweet)
        tweet = re.sub(r"shouldn't", "should not", tweet)
        tweet = re.sub(r"wouldn't", "would not", tweet)
        tweet = re.sub(r"i'm", "I am", tweet)
        tweet = re.sub(r"I\x89Ûªm", "I am", tweet)
        tweet = re.sub(r"I'm", "I am", tweet)
        tweet = re.sub(r"Isn't", "is not", tweet)
        tweet = re.sub(r"Here's", "Here is", tweet)
        tweet = re.sub(r"you've", "you have", tweet)
        tweet = re.sub(r"you\x89Ûªve", "you have", tweet)
        tweet = re.sub(r"we're", "we are", tweet)
        tweet = re.sub(r"what's", "what is", tweet)
        tweet = re.sub(r"couldn't", "could not", tweet)
        tweet = re.sub(r"we've", "we have", tweet)
        tweet = re.sub(r"it\x89Ûªs", "it is", tweet)
        tweet = re.sub(r"doesn\x89Ûªt", "does not", tweet)
        tweet = re.sub(r"It\x89Ûªs", "It is", tweet)
        tweet = re.sub(r"Here\x89Ûªs", "Here is", tweet)
        tweet = re.sub(r"who's", "who is", tweet)
        tweet = re.sub(r"I\x89Ûªve", "I have", tweet)
        tweet = re.sub(r"y'all", "you all", tweet)
        tweet = re.sub(r"can\x89Ûªt", "cannot", tweet)
        tweet = re.sub(r"would've", "would have", tweet)
        tweet = re.sub(r"it'll", "it will", tweet)
        tweet = re.sub(r"we'll", "we will", tweet)
        tweet = re.sub(r"wouldn\x89Ûªt", "would not", tweet)
        tweet = re.sub(r"We've", "We have", tweet)
        tweet = re.sub(r"he'll", "he will", tweet)
        tweet = re.sub(r"Y'all", "You all", tweet)
        tweet = re.sub(r"Weren't", "Were not", tweet)
        tweet = re.sub(r"Didn't", "Did not", tweet)
        tweet = re.sub(r"they'll", "they will", tweet)
        tweet = re.sub(r"they'd", "they would", tweet)
        tweet = re.sub(r"DON'T", "DO NOT", tweet)
        tweet = re.sub(r"That\x89Ûªs", "That is", tweet)
        tweet = re.sub(r"they've", "they have", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"should've", "should have", tweet)
        tweet = re.sub(r"You\x89Ûªre", "You are", tweet)
        tweet = re.sub(r"where's", "where is", tweet)
        tweet = re.sub(r"Don\x89Ûªt", "Do not", tweet)
        tweet = re.sub(r"we'd", "we would", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"weren't", "were not", tweet)
        tweet = re.sub(r"They're", "They are", tweet)
        tweet = re.sub(r"Can\x89Ûªt", "Cannot", tweet)
        tweet = re.sub(r"you\x89Ûªll", "you will", tweet)
        tweet = re.sub(r"I\x89Ûªd", "I would", tweet)
        tweet = re.sub(r"let's", "let us", tweet)
        tweet = re.sub(r"it's", "it is", tweet)
        tweet = re.sub(r"can't", "cannot", tweet)
        tweet = re.sub(r"don't", "do not", tweet)
        tweet = re.sub(r"you're", "you are", tweet)
        tweet = re.sub(r"i've", "I have", tweet)
        tweet = re.sub(r"that's", "that is", tweet)
        tweet = re.sub(r"i'll", "I will", tweet)
        tweet = re.sub(r"doesn't", "does not", tweet)
        tweet = re.sub(r"i'd", "I would", tweet)
        tweet = re.sub(r"didn't", "did not", tweet)
        tweet = re.sub(r"ain't", "am not", tweet)
        tweet = re.sub(r"you'll", "you will", tweet)
        tweet = re.sub(r"I've", "I have", tweet)
        tweet = re.sub(r"Don't", "do not", tweet)
        tweet = re.sub(r"I'll", "I will", tweet)
        tweet = re.sub(r"I'd", "I would", tweet)
        tweet = re.sub(r"Let's", "Let us", tweet)
        tweet = re.sub(r"you'd", "You would", tweet)
        tweet = re.sub(r"It's", "It is", tweet)
        tweet = re.sub(r"Ain't", "am not", tweet)
        tweet = re.sub(r"Haven't", "Have not", tweet)
        tweet = re.sub(r"Could've", "Could have", tweet)
        tweet = re.sub(r"youve", "you have", tweet)
        tweet = re.sub(r"donå«t", "do not", tweet)

        tweet = re.sub(r"some1", "someone", tweet)
        tweet = re.sub(r"yrs", "years", tweet)
        tweet = re.sub(r"hrs", "hours", tweet)
        tweet = re.sub(r"2morow|2moro", "tomorrow", tweet)
        tweet = re.sub(r"2day", "today", tweet)
        tweet = re.sub(r"4got|4gotten", "forget", tweet)
        tweet = re.sub(r"b-day|bday", "b-day", tweet)
        tweet = re.sub(r"mother's", "mother", tweet)
        tweet = re.sub(r"mom's", "mom", tweet)
        tweet = re.sub(r"dad's", "dad", tweet)
        tweet = re.sub(r"hahah|hahaha|hahahaha", "haha", tweet)
        tweet = re.sub(r"lmao|lolz|rofl", "lol", tweet)
        tweet = re.sub(r"thanx|thnx", "thanks", tweet)
        tweet = re.sub(r"goood", "good", tweet)
        tweet = re.sub(r"some1", "someone", tweet)
        tweet = re.sub(r"some1", "someone", tweet)
        tweet = tweet.lower()
        tweet = tweet[1:]

        # Remove all URLs
        tweet = re.sub(urlPattern, '', tweet)

        # Remove all usernames
        tweet = re.sub(userPattern, '', tweet)

        # Remove specific words
        tweet = re.sub(some, '', tweet)

        # Remove punctuation
        tweet = tweet.translate(str.maketrans("", "", string.punctuation))

        # Toekenise words
        tokens = word_tokenize(tweet)

        # Remove stop words
        final_tokens = [w for w in tokens if w not in stopword]

        # Reduce words to their stem
        wordLemm = WordNetLemmatizer()
        finalwords = []

        # Execute above
        for w in final_tokens:
            if len(w) > 1:
                word = wordLemm.lemmatize(w)
                finalwords.append(word)
        return ' '.join(finalwords)


    # Expand abreviations
    abbreviations = {
        "$": " dollar ",
        "€": " euro ",
        "4ao": "for adults only",
        "a.m": "before midday",
        "a3": "anytime anywhere anyplace",
        "aamof": "as a matter of fact",
        "acct": "account",
        "adih": "another day in hell",
        "afaic": "as far as i am concerned",
        "afaict": "as far as i can tell",
        "afaik": "as far as i know",
        "afair": "as far as i remember",
        "afk": "away from keyboard",
        "app": "application",
        "approx": "approximately",
        "apps": "applications",
        "asap": "as soon as possible",
        "asl": "age, sex, location",
        "atk": "at the keyboard",
        "ave.": "avenue",
        "aymm": "are you my mother",
        "ayor": "at your own risk",
        "b&b": "bed and breakfast",
        "b+b": "bed and breakfast",
        "b.c": "before christ",
        "b2b": "business to business",
        "b2c": "business to customer",
        "b4": "before",
        "b4n": "bye for now",
        "b@u": "back at you",
        "bae": "before anyone else",
        "bak": "back at keyboard",
        "bbbg": "bye bye be good",
        "bbc": "british broadcasting corporation",
        "bbias": "be back in a second",
        "bbl": "be back later",
        "bbs": "be back soon",
        "be4": "before",
        "bfn": "bye for now",
        "blvd": "boulevard",
        "bout": "about",
        "brb": "be right back",
        "bros": "brothers",
        "brt": "be right there",
        "bsaaw": "big smile and a wink",
        "btw": "by the way",
        "bwl": "bursting with laughter",
        "c/o": "care of",
        "cet": "central european time",
        "cf": "compare",
        "cia": "central intelligence agency",
        "csl": "can not stop laughing",
        "cu": "see you",
        "cul8r": "see you later",
        "cv": "curriculum vitae",
        "cwot": "complete waste of time",
        "cya": "see you",
        "cyt": "see you tomorrow",
        "dae": "does anyone else",
        "dbmib": "do not bother me i am busy",
        "diy": "do it yourself",
        "dm": "direct message",
        "dwh": "during work hours",
        "e123": "easy as one two three",
        "eet": "eastern european time",
        "eg": "example",
        "embm": "early morning business meeting",
        "encl": "enclosed",
        "encl.": "enclosed",
        "etc": "and so on",
        "faq": "frequently asked questions",
        "fawc": "for anyone who cares",
        "fb": "facebook",
        "fc": "fingers crossed",
        "fig": "figure",
        "fimh": "forever in my heart",
        "ft.": "feet",
        "ft": "featuring",
        "ftl": "for the loss",
        "ftw": "for the win",
        "fwiw": "for what it is worth",
        "fyi": "for your information",
        "g9": "genius",
        "gahoy": "get a hold of yourself",
        "gal": "get a life",
        "gcse": "general certificate of secondary education",
        "gfn": "gone for now",
        "gg": "good game",
        "gl": "good luck",
        "glhf": "good luck have fun",
        "gmt": "greenwich mean time",
        "gmta": "great minds think alike",
        "gn": "good night",
        "g.o.a.t": "greatest of all time",
        "goat": "greatest of all time",
        "goi": "get over it",
        "gps": "global positioning system",
        "gr8": "great",
        "gratz": "congratulations",
        "gyal": "girl",
        "h&c": "hot and cold",
        "hp": "horsepower",
        "hr": "hour",
        "hrh": "his royal highness",
        "ht": "height",
        "ibrb": "i will be right back",
        "ic": "i see",
        "icq": "i seek you",
        "icymi": "in case you missed it",
        "idc": "i do not care",
        "idgadf": "i do not give a damn fuck",
        "idgaf": "i do not give a fuck",
        "idk": "i do not know",
        "ie": "that is",
        "i.e": "that is",
        "ifyp": "i feel your pain",
        "IG": "instagram",
        "iirc": "if i remember correctly",
        "ilu": "i love you",
        "ily": "i love you",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "imu": "i miss you",
        "iow": "in other words",
        "irl": "in real life",
        "j4f": "just for fun",
        "jic": "just in case",
        "jk": "just kidding",
        "jsyk": "just so you know",
        "l8r": "later",
        "lb": "pound",
        "lbs": "pounds",
        "ldr": "long distance relationship",
        "lmao": "laugh my ass off",
        "lmfao": "laugh my fucking ass off",
        "lol": "laughing out loud",
        "ltd": "limited",
        "ltns": "long time no see",
        "m8": "mate",
        "mf": "motherfucker",
        "mfs": "motherfuckers",
        "mfw": "my face when",
        "mofo": "motherfucker",
        "mph": "miles per hour",
        "mr": "mister",
        "mrw": "my reaction when",
        "ms": "miss",
        "mte": "my thoughts exactly",
        "nagi": "not a good idea",
        "nbc": "national broadcasting company",
        "nbd": "not big deal",
        "nfs": "not for sale",
        "ngl": "not going to lie",
        "nhs": "national health service",
        "nrn": "no reply necessary",
        "nsfl": "not safe for life",
        "nsfw": "not safe for work",
        "nth": "nice to have",
        "nvr": "never",
        "nyc": "new york city",
        "oc": "original content",
        "og": "original",
        "ohp": "overhead projector",
        "oic": "oh i see",
        "omdb": "over my dead body",
        "omg": "oh my god",
        "omw": "on my way",
        "p.a": "per annum",
        "p.m": "after midday",
        "pm": "prime minister",
        "poc": "people of color",
        "pov": "point of view",
        "pp": "pages",
        "ppl": "people",
        "prw": "parents are watching",
        "ps": "postscript",
        "pt": "point",
        "ptb": "please text back",
        "pto": "please turn over",
        "qpsa": "what happens",
        "ratchet": "rude",
        "rbtl": "read between the lines",
        "rlrt": "real life retweet",
        "rofl": "rolling on the floor laughing",
        "roflol": "rolling on the floor laughing out loud",
        "rotflmao": "rolling on the floor laughing my ass off",
        "rt": "retweet",
        "ruok": "are you ok",
        "sfw": "safe for work",
        "sk8": "skate",
        "smh": "shake my head",
        "sq": "square",
        "srsly": "seriously",
        "ssdd": "same stuff different day",
        "tbh": "to be honest",
        "tbs": "tablespooful",
        "tbsp": "tablespooful",
        "tfw": "that feeling when",
        "thks": "thank you",
        "tho": "though",
        "thx": "thank you",
        "tia": "thanks in advance",
        "til": "today i learned",
        "tl;dr": "too long i did not read",
        "tldr": "too long i did not read",
        "tmb": "tweet me back",
        "tntl": "trying not to laugh",
        "ttyl": "talk to you later",
        "u": "you",
        "u2": "you too",
        "u4e": "yours for ever",
        "utc": "coordinated universal time",
        "w/": "with",
        "w/o": "without",
        "w8": "wait",
        "wassup": "what is up",
        "wb": "welcome back",
        "wtf": "what the fuck",
        "wtg": "way to go",
        "wtpa": "where the party at",
        "wuf": "where are you from",
        "wuzup": "what is up",
        "wywh": "wish you were here",
        "yd": "yard",
        "ygtr": "you got that right",
        "ynk": "you never know",
        "zzz": "sleeping bored and tired"
    }

    # Expand abreviation expansion
    print("Expanding abreviations")
    def convert_abbrev_in_text(tweet):
        t = []
        words = tweet.split()
        t = [abbreviations[w.lower()] if w.lower() in abbreviations.keys()
            else w for w in words]
        return ' '.join(t)


    # Apply lamda
    print("Applying lamdas to process tweets")
    tweets['processed_tweets'] = tweets['tweet'].apply(lambda x: process_tweets(x))
    tweets['processed_tweets'] = tweets['processed_tweets'].apply(
        lambda x: convert_abbrev_in_text(x))

    # Remove shortwords
    print("Removing shortwords")
    tweets['processed_tweets'] = tweets['processed_tweets'].apply(
        lambda x: " ".join([w for w in x.split() if len(w) > 3]))

    # Shuffle dataset
    print("Shuffling dataset")
    tweets = shuffle(tweets).reset_index(drop=True)

    # Tokenise dataset
    print("Tokenising dataset")
    tokenized_tweet = tweets['processed_tweets'].apply(lambda x: x.split())
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english', ngram_range=(
        1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(tweets['processed_tweets'].values.astype('U'))

    # Train/Test split
    X = text_counts
    y = tweets['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=19)

    # Bayes classifier
    print("Training bayes classifier")
    cnb = ComplementNB()
    cnb.fit(X_train, y_train)
    cross_cnb = cross_val_score(cnb, X, y,n_jobs = -1)
    print("\nCross Validation score = ",cross_cnb)                
    print ("Train accuracy = {:.2f}%".format(cnb.score(X_train,y_train)*100))
    print ("Test accuracy = {:.2f}%".format(cnb.score(X_test,y_test)*100))
    train_acc_cnb=cnb.score(X_train,y_train)
    test_acc_cnb=cnb.score(X_test,y_test)

    # Cache
    print("Caching classifier")

    cache = [cnb, cv]
    dump(cache, 'bayes.joblib')

def classify():
    print("Checking for already trained classifier")
    try:
        cache = load('bayes.joblib')
        print("Classifier found")
    except:
        print("Classifier not found, training")
        train()
        classify()
    
    text = "do you know what, i really like dogs"
    text = text.split(" ")
    text_features = cache[1].transform(text)
    print(text_features)
    create_word_fe
    print(cache[0].predict(text_features))

classify()