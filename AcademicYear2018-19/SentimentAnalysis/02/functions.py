import re
import os
import ast
import string
import logging  # Setting up the loggings to monitor gensim
import pickle
import multiprocessing

from time import time
from string import digits
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

# Responsible for the cleaning of the tweets
# It removes special characters (such as '#','@') and strings (such as links),
# gets rid of emoticons and stopwords
def clean_tweets(tweet):

    # Creating the list of stopwords
    stop_words = stopwords.words('english')

    extra_stop_words = ["a","about","above","after","again","against","all","am",
    "an","and","any","are","aren't","as","at","be","because","been","before",
    "being","below","between","both","but","by","can't","cannot","could",
    "couldn't","did","didn't","do","does","doesn't","doing","don't","down",
    "during","each","few","for","from","further","had","hadn't","has","hasn't",
    "have","haven't","having","he","he'd","he'll","he's","her","here","here's",
    "hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm",
    "i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on","once",
    "only","or","other","ought","our","ours","ourselves","out","over","own","same",
    "shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
    "such","than","that","that's","the","their","theirs","them","themselves",
    "then","there","there's","these","they","they'd","they'll","they're","they've",
    "this","those","through","to","too","under","until","up","very","was","wasn't",
    "we","we'd","we'll","we're","we've","were","weren't","what","what's","when",
    "when's","where","where's","which","while","who","who's","whom","why","why's",
    "with","won't","would","wouldn't","you","you'd","you'll","you're","you've",
    "your","yours","yourself","yourselve","th","im","i'm","im","rt"]

    # Adding the extra stopwords
    for i in extra_stop_words:
        if i not in stop_words:
            stop_words.append(i)

    # The stemmer we'll be using
    stemmer = SnowballStemmer('english')

    # A set containg most emoticons
    emoticons = set([':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>',
    '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D',
    'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
    ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b',
    '>:)', '>;)', '>:-)', '<3',':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(',
    ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<',
    ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';(', '(:','):','. .',
    '..',':/','::',':d'])

    i = 0
    indexes_to_drop = []
    while i < len(tweet.index):
        print("[",i,"]")

        # Removing stock market tickers like $GE
        tweet.loc[i]['Tweet'] = re.sub(r'\$\w*', '', tweet.loc[i]['Tweet'])

        # Removing old style retweet text "RT"
        tweet.loc[i]['Tweet'] = re.sub(r'^RT[\s]+', '', tweet.loc[i]['Tweet'])

        # Removing Links
        tweet.loc[i]['Tweet'] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet.loc[i]['Tweet'])

        # Removing the hashtag sign (only from the word)
        tweet.loc[i]['Tweet'] = re.sub(r'#', '', tweet.loc[i]['Tweet'])

        # Removing digits
        tweet.loc[i]['Tweet'] = ''.join([i for i in tweet.loc[i]['Tweet'] if not i.isdigit()])

        # Removing emojis
        tweet.loc[i]['Tweet'] = tweet.loc[i]['Tweet'].encode('ascii', 'ignore').decode('ascii')

        # Removing punctuation
        tweet.loc[i]['Tweet'] = re.sub(r'[^\w\s]','',tweet.loc[i]['Tweet'])

        # Tokenization
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet.loc[i]['Tweet'])
        # print(tweet_tokens)

        tweets_clean = []
        for token in tweet_tokens:
            if token in stop_words:
                # print(token," is a stopword")
                continue
            if token in emoticons:
                # print(token," is an emoticon")
                continue
            stem_word = stemmer.stem(token)
            tweets_clean.append(stem_word)
        # print(tweets_clean)

        if len(tweets_clean) == 0:
            indexes_to_drop.append(i)
        else:
            tweet.loc[i]['Tweet'] = tweets_clean
        i = i + 1

    return indexes_to_drop

# Reads tokenized data from a DataFrame object and turns them into a string
# (which will later be used for the creation of a WordCloud image)
def analyze_data(indication,df):
    comment_words = ' '

    if indication == "all":
        j = 0
        # Iterating through the 'Tweet' column of the .tsv file
        for val in df['Tweet']:
            print("[",j,"]")
            # val is read as a string, therefore we have to convert it to a list
            val = ast.literal_eval(val)

            # Creating a string from the tokens
            for words in val:
                comment_words = comment_words + words + ' '

            j = j + 1
    else:
        i = 0
        while i < len(df.index):
            print("[",i,"]")
            if df.loc[i]['Emotion'] == indication:
                val = ast.literal_eval(df.loc[i]['Tweet'])

                # Creating a string from the tokens
                for words in val:
                    comment_words = comment_words + words + ' '

            i = i + 1

    return comment_words

# Prepared the data for feature extraction
# by turning lists of tokens into strings
def prepare_data(df):
    corpus = []
    # A list of strings will be created.
    # Each string is created by combing the tokens of each row of the data set
    for val in df['Tweet']:
        # val is read as a string, therefore we have to convert it to a list
        val = ast.literal_eval(val)

        # Creating a string from the tokens
        comment_words = ' '
        for words in val:
            comment_words = comment_words + words + ' '

        # Adding the newly created string to the list
        corpus.append(comment_words)

    return corpus

# Uses the Word2Vec package from the Gensim library.
def word2vector(df,file_name):
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt= '%H:%M:%S', level=logging.INFO)

    # Gensim's word2vec expects a sequence of sentences as its input
    # Each sentence is a list of words
    sentences = []
    for val in df['Tweet']:

        # Turning the string representing the list into an actual list
        val = ast.literal_eval(val)
        sentences.append(val)

    ##################################
    #      Training of the model     #
    ##################################

    ######################
    # Step.1: Word2Vec() #
    ######################

    # Count the number of cores in the computer
    cores = multiprocessing.cpu_count()
    # print(cores)

    # Setting up the parameters of the model one by one
    # We leave the model uninitialized (that is we don't supply the
    # sentences parameter on purpose)
    w2v_model = Word2Vec(min_count=20, # (int) Ignore all words with total absolute frequency lower than this
    window=2, # (int) The maximum distance between the current and predicted word within a sentence
    size=300, # (int) Dimensionality of the feature vectors
    sample=6e-5, # (float) The threshold for configuring which higher-frequency words are randomly downsampled
    alpha=0.03, # (float) The initial learning rate
    min_alpha=0.0007, # (float) Learning rate will linearly drop to this as training progresses
    negative=20,
    workers=cores-1) # (int) Use these many worker threads to train the  model

    ##########################
    # Step.2: .build_vocab() #
    ##########################
    t = time()

    # Word2Vec requires us to build the vocabulary table
    # (simply digesting all the words and filtering out the unique words,
    # and doing some basic counts on them):
    w2v_model.build_vocab(sentences, progress_per=10000)
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    ####################
    # Step.3: .train() #
    ####################
    t = time()

    w2v_model.train(sentences,
    total_examples=w2v_model.corpus_count, # (int) count of sentences
    epochs=30, # (int) number of iterations (epochs) over the corpus
    report_delay=1)
    # print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # Since we are not planning to train the model any further, we'll call
    # init_sims(), which will make the model much more memory efficient
    w2v_model.init_sims(replace=True)

    # Turning the model's vocabulary into a list
    # model_vocabulary = []
    # for key in w2v_model.wv.vocab:
    #     model_vocabulary.append(key)
    # model_vocabulary.sort()

    #################################
    #      Exploring the model      #
    #################################

    # Finding similar words
    # print("\n",w2v_model.wv.most_similar(positive=["trump"]))

    # Finding how similar are two words to each other
    # print("\n",w2v_model.wv.similarity('obama', 'trump'))

    # Finding the odd one out
    # print("\n",w2v_model.wv.doesnt_match(['obama', 'trump', 'dog']))

    # Analogy difference
    # print("\n",w2v_model.wv.most_similar(positive=["woman", "obama"], negative=["clinton"], topn=3))

    # Vector representation of a word
    # print(w2v_model['food'])
    # for word in model_vocabulary:
        # print("\n",word,"\n",w2v_model[word],"\n")

    ##############################
    #     Storing the model      #
    ##############################
    # w2v_model.save('/tmp/mymodel')

    # Saving to a .pkl file
    pkl_path = os.getcwd() + '/pkl/' # the directory where we'll store the files
    output = open(pkl_path+file_name+'_word2vec.pkl','wb')
    pickle.dump(w2v_model,output)
    output.close()

# Converts a sentimental dictionary into a python dictionary
def txt_to_dict(filename):

    path = os.getcwd() + '/DataMining/lexica/' + filename

    f = open(path,'r') # Open the file for reading
    contents = f.readlines() # Getting the content of the file
    f.close() # Close the file

    dict = {} # Where we'll store the contents of the file

    # Reading the file line by line
    for line in contents:

        # Creating a list which holds the words of the line
        lst = line.split()

        # Skipping phrases
        if(len(lst) != 2):
            continue
        else:
            dict[lst[0]] = float(lst[1])

    return dict
