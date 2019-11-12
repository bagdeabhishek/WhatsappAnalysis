import pandas as pd
from datetime import datetime
from emoji import UNICODE_EMOJI
from tqdm import tqdm
import logging
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib as plt
import nltk
import seaborn as sns
from functools import reduce
import networkx as nx

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

logging.basicConfig(filename="log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def get_data(path="/content/drive/My Drive/colab_data/WhatsApp Chat with YJHD 😂.txt"):
    """
    This is utility funcion to read the WhatsApp Chats and extract relevant data
    :param path: The path of the WhatsApp Chat
    :type path: String
    :return: A pandas Dataframe of all the relevant information
    :rtype: pd.DataFrame
    """
    ls_rows = []
    try:
        with open(path) as f:
            for line in tqdm(f):
                message_from = None
                message_text = None
                media = False
                emojis = []
                clean_text = ""
                mention = None
                list_to_exclude = ["https",
                                   "This message was deleted",
                                   "<Media omitted>"]
                split_line = line.split(" - ")
                try:
                    date = datetime.strptime(split_line[0], "%d/%m/%y, %H:%M")
                except ValueError as e:
                    logging.debug("Not a Date: " + split_line[0] + " Exception: " + str(e))
                    continue
                message_split = split_line[1].split(":")
                if len(message_split) > 1:
                    message_from = message_split[0]
                    message_text = message_split[1].strip()
                    if "<Media omitted>" in message_text:
                        media = True
                    if any(exclude in message_text for exclude in list_to_exclude):
                        message_text = None
                    else:
                        if "@" in message_text:
                            new_message = ""
                            for word in message_text.split():
                                if word.startswith("@"):
                                    mention = word
                                    continue
                                new_message += word
                            message_text = new_message
                        for character in message_text:
                            if character in UNICODE_EMOJI:
                                emojis.append(character)
                            else:
                                clean_text += character
                clean_text = None if clean_text.strip() == "" else clean_text
                emojis = None if len(emojis) < 1 else ','.join(emojis)
                POS = __get_relevant_words(clean_text)
                ls_rows.append((date, message_from, message_text, media, emojis, clean_text, mention, POS))
        df = pd.DataFrame(ls_rows, columns=["time", "from", "text", "media", "emojis", "clean_text", "mention", "POS"])
        df.dropna(subset=['text'], inplace=True)
        return df
    except Exception as e:
        print("Critical Exception " + str(e))
        return


def __get_relevant_words(sentence):
    """
    Extracts words which are Nouns or Foreign Words only. Mostly relevant for clean Word cloud
    :param sentence: This sentence from which to extract relevant words
    :type sentence: String
    :return: A string of relevant words
    :rtype: String
    """
    nouns = None
    try:
        if sentence:
            tokens = nltk.word_tokenize(sentence)
            pos = nltk.pos_tag(tokens)
            nouns = [x[0] for x in pos if x[1].startswith('N') or x[1].startswith('F')]
    except Exception as e:
        nouns = None
    return ' '.join(nouns) if nouns else None


def get_word_freq_dict(df_col):
    """
    Get word frequency dictionary from a DataFrame Column
    :param df_col: The column from which to generate word frequency dictionary
    :type df_col: DataFrame
    :return: Dictionary where key is the word and value is the frequency
    :rtype: Dict
    """
    results = Counter()
    df_col.str.lower().str.split().apply(results.update)
    results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    d = {}
    for word, freq in results:
        d[word] = freq
    return d


def plot_word_cloud(word_freq_dict, stopwords=STOPWORDS, background_color="white", width=800, height=1000,
                    max_words=300, figsize=(50, 50)):
    """
    Display the Word Cloud using Matplotlib
    :param word_freq_dict: Dictionary of word frequencies
    :type word_freq_dict: Dict
    :return: None
    :rtype: None
    """
    word_cloud = WordCloud(stopwords=stopwords, background_color=background_color, width=width, height=height,
                           max_words=max_words).generate_from_frequencies(frequencies=word_freq_dict)
    plt.figure(figsize=figsize)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def top_emojis(df, name):
    """
    Get the top emojis used by a user. (NO USE NOW)
    :param df: The Dataframe with user name and emoji
    :type df: DataFrame
    :param name: Name of the user for which to find the top emojis used
    :type name: String
    :return: list of tuples with (emoji, frequency in the chat) in sorted order
    :rtype: list
    """
    counter = Counter()
    df.loc[df["from"] == name]["emojis"].str.split(",").apply(counter.update)
    counter = (sorted(counter.items(), key=lambda x: x[1], reverse=True))
    return counter


def clean_data(df, new_name=None):
    """
    Clean the given data and perform basic cleaning operations
    :param df: The DataFrame extracted from given whatsapp chats
    :type df: DataFrame
    :param new_name: list of names if you want to replace senders name to something shorter
    :type new_name: List
    :return: Cleaned DataFrame
    :rtype: DataFrame
    """

    if new_name:
        original_name = df["from"].unique().tolist()
        df.replace(original_name, new_name, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    df.set_index('time', inplace=True, drop=False)
    return df


def plot_message_counts(df, size=(20, 4), freq='d'):
    """
    Get the statistics of messages sampled by day
    :param freq: String representing the sampling frequencies (refer:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
    :type freq: String
    :param size: Size of the generated plot
    :type size: tuple(height,width)
    :param df: The extracted Dataframe
    :type df: DataFrame
    :return:
    :rtype:
    """
    df_resampled = df.resample(freq)
    sns.set(rc={'figure.figsize': size})
    df_resampled.count().plot(linewidth=0.5)


def group_by_time(df, period='day'):
    """
    Group the whole data by a time period and return the description
    :param df:
    :type df:
    :param period:
    :type period:
    :return:
    :rtype:
    """
    description = df.groupby(df.time.dt.day).describe()
    ls_from = description['from']
    ls_text = description['cleaned_text']
    plot_emoji_heatmap(df, agg=df.time.dt.day)  # Plot daily emoji use heat map


def plot_emoji_heatmap(df, size=(20, 5), agg='from'):
    df_smiley = df.groupby(agg)['emojis'].agg(['count', __custom_smiley_aggregator])
    ls_smiley = []
    for x in df_smiley.itertuples():
        for smiley, count in x._2:
            ls_smiley.append((x.Index, smiley, count))
    df_smiley_reduced = pd.DataFrame(ls_smiley, columns=["agg", "smiley", "count"])
    df_smiley_reduced = df_smiley_reduced.pivot_table('count', ['agg'], 'smiley').fillna(0)
    sns.set(rc={'figure.figsize': size})
    sns.heatmap(df_smiley_reduced.transpose(), cmap="Blues")


def __custom_smiley_aggregator(series):
    c = Counter()
    for x in series.tolist():
        if x:
            for smiley in x.split(','):
                c.update(smiley)
    return c.most_common(5)
