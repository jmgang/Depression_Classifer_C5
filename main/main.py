import twitter
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from translate import Translator
from pprint import pprint


translator = Translator(from_lang="tagalog", to_lang="english")

def authenticate_twitter():
    # initialize api instance
    twitter_api = twitter.Api(consumer_key = "",
                                consumer_secret = "",
                                access_token_key = "",
                                access_token_secret = "")
    # test authentication
    print(twitter_api.VerifyCredentials())

    return twitter_api

def pull_tweets_from_twitter(twitter_api, count_of_tweets, search_keyword='twitter'):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count=count_of_tweets, lang='en')
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

def pull_tweets_from_user_twitter(twitter_api, count_of_tweets, username):
    try:
        tweets_fetched = twitter_api.GetUserTimeline(screen_name=username, count=100)
        print("Fetched " + str(len(tweets_fetched)))
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None

def clean_twitter_data(tweet_data):
    stop_words = get_stop_words_list('stop_words.txt')
    cleaned_tweets = []
    all_words = []
    tweet_data_dict = [{}]
    # tweet_data = set(tweet["tweet"] for tweet in tweet_data)
    # print_list(tweet_data, 'tweet_data')

    for tweet in tweet_data:
        #split tweet into words
        words = tweet["tweet"].split()
        single_tweet = []
        for w in words:
            #strip punctuation
            w = w.strip('\'"?,.')
            #check if the word stats with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            #ignore if it is a stop word
            if(w in stop_words or val is None):
                continue
            else:
                #translate
                token = translator.translate(w.lower())
                single_tweet.append(token)
                all_words.append(token)
        cleaned_tweets.append(single_tweet)
        tweet["cleaned"] = single_tweet
    return cleaned_tweets, all_words, tweet_data

def retrieve_tweets_from_file(data_file='tweets.csv'):
    retrieved_tweets = []
    ctr = 1
    with open(data_file, mode='r', encoding="utf8") as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
            if row is not None and row != []:
                retrieved_tweets.append({"id": ctr, "tweet" : row[1], "A_Attr": row[2], "S_Attr" : row[3], "D_Label":row[4]})
                ctr += 1
    # print_list(retrieved_tweets)
    return retrieved_tweets


def get_stop_words_list(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

def get_words_list_from_file(from_file):
    #read the from file and build a list
    words_list = []

    fp = open(from_file, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        words_list.append(word)
        line = fp.readline()
    fp.close()
    return words_list

def get_all_words_from_tweets(tweet_data):
    lst = []
    for cleaned_tweet in tweet_data:
        lst.extend(cleaned_tweet["cleaned"])
    return list(set(lst))

def build_BOW_C5(tweet_data):
    all_words = get_all_words_from_tweets(tweet_data)

    for tweet in tweet_data:
        bow_list = []
        for word in tweet["cleaned"]:
            pass
    return all_words


def build_rule_based_system(tweet_data, from_file):
    bag_of_words = list(set(get_words_list_from_file(from_file)))
    bag_of_words_list = {}

    for word in bag_of_words:
        bag_of_words_list[word] = 0

    for tweet in tweet_data:
        for word in bag_of_words:
            bag_of_words_list[word] = count_word(word, tweet["cleaned"])
        sum = 0
        for word in bag_of_words_list:
            sum += bag_of_words_list[word]
        tweet["bow_sum"] = sum

    return tweet_data


def count_word(find, sentence):
    ctr = 0
    for word in sentence:
        if find == word:
            ctr += 1
    return ctr



def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def print_list(lst, title=''):
    if title != '':
        print(title)
    for k in lst:
        print(k)
    print()

def print_list_with_index(lst):
    for idx, k in enumerate(lst):
        print((idx+1), k)
    print()

def find_active(tweet_data):
    tweets = []
    is_active = False
    for tweet in tweet_data:
        for keyword in tweet["bag_of_words"]:
            if tweet["bag_of_words"][keyword] > 0:
                tweets.append(tweet)
                break
    return tweets



def to_list(bag_of_words):
    lst = []

    for keyword in bag_of_words:
        lst.append(bag_of_words[keyword])

    return lst

def write_bag_of_words_to_CSV(tweet_data, tweetDataFile='bag_of_words.csv'):
     # Now we write them to the empty CSV file
    with open(tweetDataFile, mode='w', encoding="utf8") as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        headers = []
        headers.append("Tweet ID")
        headers.append("Cleaned Tweet")

        keywords = list(set(get_words_list_from_file(from_file='loss_of_energy.txt')))
        # keywords.sort()
        headers.extend(keywords)
        headers.append("Symptom")
        linewriter.writerow(headers)


        for tweet in tweet_data:
            try:
                lst = []
                lst.append(tweet["id"])
                lst.append(tweet["cleaned"])
                lst.extend(to_list(tweet["bag_of_words"]))
                lst.append("Loss of Energy")
                linewriter.writerow(lst)
            except Exception as e:
                print(e)

def get_word_weights():
    word_weights = {}
    filename = 'word_weights.txt'
    with open(filename, 'r') as file_read:
        lines = file_read.readlines()
        for line in lines:
            str_lst = line.split()
            weight = str_lst[1]
            try:
                weight = float(weight)
            except ValueError:
                weight = 0

            word_weights[str_lst[0]] = weight
    return word_weights


def assign_scores_to_tweets(tweet_data):
    word_weights = get_word_weights()
    for tweet in tweet_data:
        score = 0
        norm = 0
        total_words = len(tweet["cleaned"])
        for token in tweet["cleaned"]:
            if token in word_weights:
                weight = word_weights[token]
                if(weight < 0):
                    norm += 1
                score += weight
        tweet["score"] = score
        tweet["norm"] = norm
        total_words = total_words - norm

        if total_words > norm:
            tweet["total_words"] = "more"
        elif total_words < norm:
            tweet["total_words"] = "less"
        else:
            tweet["total_words"] = "equal"

    return tweet_data

def write_to_CSV(tweets, tweetDataFile='tweets.csv'):
     # Now we write them to the empty CSV file
    with open(tweetDataFile, mode='w', encoding="utf8") as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in tweets:
            try:
                linewriter.writerow([tweet["text"], tweet["label"]])
            except Exception as e:
                print(e)


def extract_list_elements_to_string(lst):
    new_lst = []
    for element in lst:
        new_lst.append(element[0])
    return new_lst


# X = dataset.iloc[:, 1:2].values # 0 = created at. 1 = tweeets 2 = label
def build_and_count_BOWs(X, tweet_data):
    vectorizer = CountVectorizer()
    processed_features = vectorizer.fit_transform(extract_list_elements_to_string(X))

    features_list = processed_features.toarray().tolist()

    for idx, tweet in enumerate(tweet_data):
        sum = 0
        for feature in features_list[idx]:
            sum += feature
        tweet["bow_sum"] = sum
    return tweet_data

def serialize_final_tweets(tweets, tweetDataFile='final_c5_data.csv'):
    with open(tweetDataFile, mode='w', encoding="utf8", newline='') as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        linewriter.writerow(['A_Attr', 'S_Attr', 'BOW_Sum', 'Word_Score', 'Norm', 'Total_Words', 'Label'])
        for tweet in tweets:
            try:
                linewriter.writerow([tweet["A_Attr"], tweet["S_Attr"], tweet["bow_sum"], tweet["score"],
                                     tweet["norm"], tweet["total_words"], tweet["D_Label"]])
            except Exception as e:
                print(e)

    print("output file ready..")


def randomize_and_split( tweet_data_local ):
    # randomize
    from random import Random
    rand = Random()
    rand.shuffle(tweet_data_local)

    size70 = round(len(tweet_data_local) * 0.7)

    train_set = tweet_data_local[:size70]
    test_set = tweet_data_local[size70+1:]

    y_train = [ x['D_Label'] for x in train_set ]
    y_test = [ x['D_Label'] for x in test_set ]
    return train_set, y_train, y_test

def get_tweet_dict( t_data, term ):
    import collections
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    t_dict = {}
    for t in t_data:
        if compare( t['cleaned'], term ):
            t_dict = t
            break
    return t_dict

def normalize_total_words(total_words_values):
    total_words_map = {'more' : 1, 'equal' : 0.5, 'less': 0}

    for word_list in total_words_values:
        word_list[0] = total_words_map[word_list[0]]

    return total_words_values


# X = dataset.iloc[:, 1:2].values # 0 = created at. 1 = tweeets 2 = label
def naive_bayes_classifier(dataset):

    scaler = MinMaxScaler()
    normalized_word_score = scaler.fit_transform(dataset.Word_Score.values.reshape(-1,1))
    # print(normalized_word_score)

    dataset.Word_Score = normalized_word_score
    dataset.Total_Words = normalize_total_words(dataset.Total_Words.values.reshape(-1,1))

    X_values = dataset.iloc[:, 0:5].values
    y_values = dataset.iloc[:,6].values

    print(X_values)
    print(y_values)

    X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.3, random_state=0)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    print("Multinomial Score: " + str(nb_classifier.score(X_test, y_test)))

    y_pred = nb_classifier.predict(X_test)

    print(y_pred)

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test,y_pred))
    print("\nClassification Report")
    print(classification_report(y_test,y_pred))
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n")


if __name__ == '__main__':
    print('classifying tweets with depression using Naive Bayes')

    tweet_data = retrieve_tweets_from_file(data_file='final_c5_tweets.csv')
    cleaned_tweets, all_words, tweet_data = clean_twitter_data(tweet_data)

    pprint(tweet_data)

    # print('Pulled data: ')
    # print_list_with_index(tweet_data)
    #
    # print('Cleaned Tweets: ')
    # print_list_with_index(cleaned_tweets)
    #
    # print('All Words: ')
    # print_list_with_index(all_words)
    #
    # print_list(tweet_data, 'tweet data cleaned')

    # feature extraction
    # dataset = pd.read_csv('final_c5_tweets.csv')
    # tweet_data_with_bow = build_and_count_BOWs(dataset.iloc[:, 1:2].values, tweet_data[1:])
    # overall_tweet_data = assign_scores_to_tweets(tweet_data_with_bow)
    #
    # pprint(overall_tweet_data)
    #
    # pprint(overall_tweet_data)
    #

    dataset = pd.read_csv('final_c5_data.csv')

    naive_bayes_classifier(dataset)




