import twitter
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from pprint import pprint



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
                single_tweet.append(w.lower())
                all_words.append(w.lower())
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

def get_triggered_tweets(tweet_data, trigger_file='trigger_words.txt'):
    # tweet data - list of dicts

    trigger_words_list = get_words_list_from_file(trigger_file)
    triggered_tweet_data = []

    for tweet in tweet_data:
        cleaned_tweet_list = tweet["cleaned"]
        for trigger_word in trigger_words_list:
            if trigger_word in cleaned_tweet_list:
                tweet["label"] = "depressed"
                triggered_tweet_data.append(tweet)
    return triggered_tweet_data


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


def assign_class(tweet_data, min_threshold, max_threshold):
    for tweet in tweet_data:
        tweet_score = tweet["score"]
        if tweet_score > min_threshold and tweet_score < max_threshold:
            tweet["symptom"] = 1
        else:
            tweet["symptom"] = 0
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

def prepare_for_C5(tweets, tweetDataFile='final_c5_data.csv'):
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

def dfeatures(document, t_set):
    # make a CountVectorizer-style tokenizer
    tokenize = CountVectorizer().build_tokenizer()
    terms = tokenize(document)

    tdd = get_tweet_dict(t_set, terms)

    # print(str(tweet_data_dict['id']) + '\t' + str(terms))
    d = {'count': len(terms), 'score': tdd['score'], 'total_words' : tdd['total_words'], 'norm': tdd['norm'],
            'a_attr': tdd['A_Attr'], 's_attr' : tdd['S_Attr']}
    for t in terms:
        d[t] = d.get(t, 0) + 1
    return d

# X = dataset.iloc[:, 1:2].values # 0 = created at. 1 = tweeets 2 = label
def naive_bayes_classifier(tweet_data):

    train_set, y_train, y_test = randomize_and_split(tweet_data)
    cleaned_tweets = [ x['cleaned'] for x in train_set ]
    tweets = []
    for tweet_list in cleaned_tweets:
        tweet_string = ''
        for word in tweet_list:
            tweet_string += word + ' '
        if tweet_string:
            tweets.append(tweet_string.rstrip())
    norm = [c['norm'] for c in train_set]

    from sklearn.feature_extraction import DictVectorizer

    print_list(train_set, 'train_set')
    vect = DictVectorizer()
    X_train = vect.fit_transform(dfeatures(t, train_set) for t in tweets)

    print(X_train.toarray())
    print(X_train.shape)

    print(len(y_train))

    nb_classifier = MultinomialNB()

    nb_classifier.fit( X_train, y_train )



if __name__ == '__main__':
    print('classifying tweets with depression using C5')

    # Retrieval of Tweets (Uncomment if first time), Comment for succeeeding times
    # twitter_api = authenticate_twitter()
    # tweets = pull_tweets_from_twitter(twitter_api, 120, 'depressed')
    # print(tweets)
    # write_to_CSV(tweets)


    # NEW IMPLMENTATION - Retrieving tweets from actual user (instead of just searching for keywords)
    # twitter_api = authenticate_twitter()
    # tweets = pull_tweets_from_user_twitter(twitter_api, 100, "depressionarmy")
    # print(tweets)
    # write_to_CSV(tweets, tweetDataFile='tweets2.csv')

    tweet_data = retrieve_tweets_from_file(data_file='final_c5_tweets.csv')
    cleaned_tweets, all_words, tweet_data = clean_twitter_data(tweet_data)

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

    # triggered_tweet_data = get_triggered_tweets(tweet_data)
    # print_list(triggered_tweet_data, 'triggered tweets')

    # bow = build_BOW_C5( tweet_data)
    #
    # pprint(bow)

    dataset = pd.read_csv('final_c5_tweets.csv')
    tweet_data_with_bow = build_and_count_BOWs(dataset.iloc[:, 1:2].values, tweet_data[1:])

    overall_tweet_data = assign_scores_to_tweets(tweet_data_with_bow)

    print_list(overall_tweet_data, 'overrall tweet data')

    naive_bayes_classifier(overall_tweet_data)



    # pprint(overall_tweet_data)
    #
    # prepare_for_C5(overall_tweet_data)

    # tweet_data = build_rule_based_system(tweet_data, from_file='loss_of_energy.txt')
    # pprint(tweet_data)

    # print("Tweets with Hits with symptom of Loss of Energy")
    # actives = find_active(tweet_data)
    # pprint(actives)
    #
    # write_bag_of_words_to_CSV(tweet_data)

    #word_weights = get_word_weights()
    #pretty(word_weights)

    # tweet_data = assign_class(assign_scores_to_tweets(tweet_data), -100, 0)

    # print("cleaned tweet, score, norm, total words, symptom/label")
    # for tweet in tweet_data:
    #     print(str(tweet["cleaned"]) + "\t" + str(tweet["score"]) + "\t" + str(tweet["norm"]) + "\t" + str(tweet["total_words"]) + "\t" + str(tweet["symptom"]))

   #prepare_for_C5(tweet_data)




