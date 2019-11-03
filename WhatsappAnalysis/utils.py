import pandas as pd
from datetime import datetime
from emoji import UNICODE_EMOJI
from tqdm import tqdm
import logging
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib as plt
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


def get_word_cloud(df_col):
    """
    Get word cloud from a DataFrame Column
    :param df_col: The column from which to generate 
    :type df_col:
    :return:
    :rtype:
    """
    results = Counter()
    df_col.str.lower().str.split().apply(results.update)
    results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    stopwords = STOPWORDS
    d = {}
    for word, freq in results:
        # print(x,y)
        if len(word) > 3:
            d[word] = freq
    word_cloud = WordCloud(stopwords=stopwords, background_color="white", width=800, height=1000,
                          max_words=300).generate_from_frequencies(frequencies=d)
    plt.figure(figsize=(50, 50))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
