import twitter
import csv
import re

def authenticate_twitter():
    # initialize api instance
    twitter_api = twitter.Api(consumer_key = "OHa1YzGAA9bCYgGLqTj8Gx9lY",
                                consumer_secret = "Aidms2U04HsS4qwE1gHiuO8PtzOHoeC27GQ1a97y60eL09zYNt",
                                access_token_key = "1174489071864823813-OWzSpSAM6Or9i3OylovvdYTsH6s4OY",
                                access_token_secret = "fxVI68nKxsjxbzEVjuEo3fapFrKCWzJcSkyKhGOFkf2kY")
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


def write_to_CSV(tweets, tweetDataFile='tweets.csv'):
     # Now we write them to the empty CSV file
    with open(tweetDataFile, mode='w', encoding="utf8") as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in tweets:
            try:
                linewriter.writerow([tweet["text"], tweet["label"]])
            except Exception as e:
                print(e)

def clean_twitter_data(tweet_data):
    stop_words = get_stop_words_list('stop_words.txt')
    cleaned_tweets = []
    all_words = []
    tweet_data_dict = [{}]
    # tweet_data = set(tweet["tweet"] for tweet in tweet_data)
    print_list(tweet_data, 'tweet_data')

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
                retrieved_tweets.append({"id": ctr, "tweet" : row[0], "label":row[1]})
                ctr += 1
    print_list(retrieved_tweets)
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



    tweet_data = retrieve_tweets_from_file(data_file='tweets2.csv')

    clean_twitter_data(tweet_data)
    cleaned_tweets, all_words, tweet_data = clean_twitter_data(tweet_data)

    print('Pulled data: ')
    print_list_with_index(tweet_data)

    print('Cleaned Tweets: ')
    print_list_with_index(cleaned_tweets)

    print('All Words: ')
    print_list_with_index(all_words)

    print_list(tweet_data, 'tweet data cleaned')

    triggered_tweet_data = get_triggered_tweets(tweet_data)
    print_list(triggered_tweet_data, 'triggered tweets')
