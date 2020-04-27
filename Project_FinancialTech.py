import numpy as np
import pandas as pd
from collections import Counter
import math
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import googletrans
from googletrans import Translator
import os
from flask import (
    Flask,
    escape,
    request,
    Response,
    render_template,
    redirect,
    url_for,
    send_file,
)
import gdown
import csv
from googlesearch import search
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as senti
from textblob import TextBlob


app = Flask(__name__)

# list words for lemmatization
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
wordnet
# import dataset

cwd = os.getcwd()

# import dataset from Google Drive
url = "https://drive.google.com/uc?id=1y3h9VhQQgtvEuU0pvako1yO1RTPB86dz&export=download"
output = "dataset.csv"
gdown.download(url, output, quiet=False)

database = pd.read_csv(os.path.join(cwd, "dataset.csv"))
database.drop_duplicates(subset="Blurb", inplace=True)
database.reset_index(drop=True, inplace=True)
database["Sentiment Title"], database["Sentiment Blurb"] = np.nan, np.nan

for i in range(len(database["Title"])):
    blob = TextBlob(database["Title"].loc[i])
    database["Sentiment Title"].loc[i] = blob.sentiment.polarity

for i in range(len(database["Blurb"])):
    blob = TextBlob(database["Blurb"].loc[i])
    database["Sentiment Blurb"].loc[i] = blob.sentiment.polarity

# function to clean blurb
def blurb_cleaner(blurb):
    blurb = re.sub(r"[^\w\s\'\/]", " ", blurb)
    blurb = re.sub(r" \'", "", blurb)
    blurb = re.sub(r" +", " ", blurb)
    blurb = re.sub(r" \'", "", blurb)
    blurb = re.sub(r"^[ \t]+|[ \t]+$", "", blurb)
    return blurb


book_data_pure = database
database["Blurb"] = database["Blurb"].apply(blurb_cleaner)

# function to find blurb given the book title
def get_blurb(user_title):
    value_holder = 70
    my_blurb = ""
    for ttl in database["Title"]:
        score = fuzz.token_sort_ratio(user_title, ttl)
        if score > value_holder:
            title = ttl
            my_blurb = database["Blurb"][database["Title"] == title]
            return my_blurb
        else:
            next
    if my_blurb == "":
        my_blurb = ["ERROR"]
        return my_blurb


# vectorize blurb
def word2vec(word):
    count_characters = Counter(word)
    set_characters = set(count_characters)
    length = math.sqrt(sum(c * c for c in count_characters.values()))
    return count_characters, set_characters, length, word


def cosine_similarity(vector1, vector2, ndigits):
    common_characters = vector1[1].intersection(vector2[1])
    product_summation = sum(
        vector1[0][character] * vector2[0][character] for character in common_characters
    )
    length = vector1[2] * vector2[2]
    if length == 0:
        similarity = 0
    else:
        similarity = round(product_summation / length, ndigits)
        return similarity


# categorize words inside blurb
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


# cosine similarity between blurbs given two titles
def find_similar(title_1, title_2):
    input_1 = pd.DataFrame(get_blurb(title_1), columns=["Blurb"])
    input_2 = pd.DataFrame(get_blurb(title_2), columns=["Blurb"])

    results = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(input_1)):
        if input_1["Blurb"].iloc[i] == "ERROR":
            results = 0
        else:
            input_v1 = [
                lemmatizer.lemmatize(w, get_wordnet_pos(w))
                for w in nltk.word_tokenize(input_1["Blurb"].iloc[i])
            ]
            input_v1 = word2vec(input_v1)
            for j in range(len(input_2)):
                if input_2["Blurb"].iloc[j] == "ERROR":
                    results = 0
                else:
                    input_v2 = [
                        lemmatizer.lemmatize(w, get_wordnet_pos(w))
                        for w in nltk.word_tokenize(input_2["Blurb"].iloc[j])
                    ]
                    input_v2 = word2vec(input_v2)
                    similarity_score = cosine_similarity(input_v1, input_v2, 5)
                    results.append(similarity_score)
    return np.around(np.mean(results), 5)


# get blurb from uncleaned dataset
def get_blurb_2(user_title1):
    value_holder = 70
    my_blurb = ""
    for ttl in book_data_pure["Title"]:
        score = fuzz.token_sort_ratio(user_title1, ttl)
        if score > value_holder:
            title = ttl
            my_blurb = book_data_pure["Blurb"][book_data_pure["Title"] == title]
            return my_blurb
        else:
            next
    if my_blurb == "":
        my_blurb = ["ERROR"]
        return my_blurb


# translation
def get_trans(text, destination):
    translator = Translator()
    lang_list = list(googletrans.LANGUAGES.values())
    x = googletrans.LANGUAGES
    if destination in lang_list:
        dest = list(x.keys())[list(x.values()).index(destination)]
        output = translator.translate(text, dest=dest).text
    else:
        output = "ERROR: Please enter another language"
    return output


# get ISBN from title
def get_isbn(user_title):
    value_holder = 70
    my_isbn = ""
    for ttl in database["Title"]:
        score = fuzz.token_sort_ratio(user_title, ttl)
        if score > value_holder:
            title = ttl
            my_isbn = database["ISBN"][database["Title"] == title]
            return my_isbn
        else:
            next
    if my_isbn == "":
        my_isbn = ["ERROR"]
        return my_isbn


# sentiment analysis of blurb given book title
def find_sentiment(title_1):
    input_1 = pd.DataFrame(get_blurb(title_1), columns=["Blurb"])
    temp_key = []

    for i in range(len(input_1)):
        if input_1["Blurb"].iloc[i] == "ERROR":
            temp_key = -2
        else:
            my_score = senti().polarity_scores(input_1["Blurb"].iloc[i])
            key = my_score["compound"]
            temp_key.append(key)

    tot_key = np.mean(temp_key)
    if -1.99 <= tot_key < -0.67:
        output = "negative"
    elif -0.67 <= tot_key < 0.66:
        output = "neutral"
    elif tot_key >= 0.66:
        output = "positive"
    else:
        output = "ERROR"

    return output


# homepage
@app.route("/home", methods=["GET", "POST"])
def homepage():
    return render_template("main.html")


# page with all search services
@app.route("/search_service", methods=["GET", "POST"])
def search_service():
    return render_template("search.html")


# ------------------- - search Author - -------------------
@app.route("/search_author", methods=["GET", "POST"])
def search_author():
    render_template("search_a.html")
    if request.method == "POST":
        return redirect(url_for("/search_a_result"))
    else:
        return render_template("search_a.html")


@app.route("/search_a_result", methods=["GET", "POST"])
def search_a_result():
    user_input = request.form["text"]

    def find_au(author_input):
        output0 = database[database["Author"] == author_input][
            ["Title", "Publisher", "Year", "ISBN"]
        ]
        return output0

    auth = find_au(user_input)

    return render_template("search_solution.html", tables=auth.to_html())


# ------------------- - search Title - -------------------
@app.route("/search_title", methods=["GET", "POST"])
def search_title():
    render_template("search_t.html")
    if request.method == "POST":
        return redirect(url_for("/search_t_result"))
    else:
        return render_template("search_t.html")


@app.route("/search_t_result", methods=["GET", "POST"])
def search_t_result():
    user_input1 = request.form["text"]

    def find_ti(title_input):
        output1 = database[database["Title"] == title_input][
            ["Author", "Publisher", "Year", "ISBN"]
        ]
        return output1

    title = find_ti(user_input1)

    return render_template("search_solution.html", tables=title.to_html())


# ------------------- - search Isbn - -------------------
@app.route("/search_isbn", methods=["GET", "POST"])
def search_isbn():
    render_template("search_i.html")
    if request.method == "POST":
        return redirect(url_for("/search_i_result"))
    else:
        return render_template("search_i.html")


@app.route("/search_i_result", methods=["GET", "POST"])
def search_i_result():
    user_input2 = request.form["text"]

    def find_is(isbn_input):
        output2 = database[database["ISBN"] == int(isbn_input)][
            ["Author", "Publisher", "Title", "Year"]
        ]
        return output2

    isbn = find_is(user_input2)

    return render_template("search_solution.html", tables=isbn.to_html())


# ------------------- - search Year - -------------------
@app.route("/search_year", methods=["GET", "POST"])
def search_year():
    render_template("search_y.html")
    if request.method == "POST":
        return redirect(url_for("/search_y_result"))
    else:
        return render_template("search_y.html")


@app.route("/search_y_result", methods=["GET", "POST"])
def search_y_result():
    user_input3 = request.form["text"]

    def find_ye(year_input):
        output3 = database[database["Year"] == float(year_input)][
            ["Author", "Title", "Publisher", "ISBN"]
        ]
        return output3

    year = find_ye(user_input3)

    return render_template("search_solution.html", tables=year.to_html())


# ------------------- - search Publisher - -------------------
@app.route("/search_publisher", methods=["GET", "POST"])
def search_publisher():
    render_template("search_p.html")
    if request.method == "POST":
        return redirect(url_for("/search_p_result"))
    else:
        return render_template("search_p.html")


@app.route("/search_p_result", methods=["GET", "POST"])
def search_p_result():
    user_input4 = request.form["text"]

    def find_pu(publisher_input):
        output4 = database[database["Publisher"] == publisher_input][
            ["Author", "Title", "Year", "ISBN"]
        ]
        return output4

    publ = find_pu(user_input4)

    return render_template("search_solution.html", tables=publ.to_html())


@app.route("/similarity", methods=["GET", "POST"])
def similarity():
    render_template("find_similar.html")
    if request.method == "POST":
        return redirect(url_for("/similarity_result"))
    else:
        return render_template("find_similar.html")


@app.route("/similarity_result", methods=["GET", "POST"])
def similarity_result():
    user_title1 = request.form["text"]
    user_title2 = request.form["text2"]

    def scoring_book(user_title1, user_title2):
        score = find_similar(user_title1, user_title2)
        if score == 0:
            output = "ERROR: please check inputs"
        else:
            output = "The cosine similarity score is {}".format(score)
        return output

    output = scoring_book(user_title1, user_title2)

    return render_template("search_solution.html", tables=output)


@app.route("/translation", methods=["GET", "POST"])
def translation():
    render_template("transl.html")
    if request.method == "POST":
        return redirect(url_for("/transl_result"))
    else:
        return render_template("transl.html")


# Translate the blurb given a title in the language the user inputted
@app.route("/transl_result", methods=["GET", "POST"])
def transl_result():
    user_title1 = request.form["text"]
    user_language = request.form["text2"]

    def final_text(user_title_input, user_language):
        target = pd.DataFrame(get_blurb_2(user_title_input), columns=["Blurb"])
        output = pd.DataFrame(columns=["Translation"])
        for i in range(len(target)):
            if target["Blurb"].iloc[i] == "ERROR":
                return "ERROR: Please select another Title"
            else:
                temp_out = pd.DataFrame(
                    [get_trans(target["Blurb"].iloc[i], destination=user_language)],
                    columns=["Translation"],
                )
                output = output.append(temp_out, ignore_index=True)
        return output

    trans = final_text(user_title1, user_language)

    return render_template("search_solution.html", tables=trans.to_html())


# return useful links for ISBN
@app.route("/links_isbn", methods=["GET", "POST"])
def links_isbn():
    render_template("link_is.html")
    if request.method == "POST":
        return redirect(url_for("/link_is_result"))
    else:
        return render_template("link_is.html")


@app.route("/link_is_result", methods=["GET", "POST"])
def link_is_result():
    user_title1 = request.form["text"]

    def book_lookup(title):
        isbn = pd.DataFrame(get_isbn(title), columns=["ISBN"])
        output = pd.DataFrame(columns=["Link"])
        for i in range(len(isbn)):
            if isbn["ISBN"].iloc[i] == "ERROR":
                output["Link"].iloc[
                    i
                ] = "ERROR: Unable to find ISBN. Please enter another title"
            else:
                for x in search(
                    isbn["ISBN"].iloc[i], tld="com", num=5, stop=5, pause=2
                ):
                    output = output.append({"Link": x}, ignore_index=True)
        return output

    links = book_lookup(user_title1)

    return render_template("search_solution.html", tables=links.to_html())


@app.route("/sentiment_score", methods=["GET", "POST"])
def sentiment_score():
    render_template("senti.html")
    if request.method == "POST":
        return redirect(url_for("/sentiment_result"))
    else:
        return render_template("senti.html")


@app.route("/sentiment_result", methods=["GET", "POST"])
def sentiment_result():
    user_title1 = request.form["text"]

    def book_sentiment(title):
        sentimento = find_sentiment(title)
        if sentimento == "ERROR":
            return "ERROR: Please enter another title"
        else:
            return sentimento

    senti1 = book_sentiment(user_title1)

    return render_template("search_solution1.html", titles=user_title1, sentim=senti1)


# ------------------ Sentiment for Title ------------------
@app.route("/sentiment_title", methods=["GET", "POST"])
def sentiment_title():
    render_template("senti_t.html")
    if request.method == "POST":
        return redirect(url_for("/sentiment_title_result"))
    else:
        return render_template("senti_t.html")


@app.route("/sentiment_title_result", methods=["GET", "POST"])
def sentiment_title_result():
    user_num = request.form["Number"]

    def bysentiment_title(num):
        database.sort_values(by=["Sentiment Title"], inplace=True)

        top = database[["Title", "Sentiment Title"]][-int(num):]
        bottom = database[["Title", "Sentiment Title"]][:int(num)]

        return top, bottom

    table_senti = bysentiment_title(user_num)

    return render_template("search_solution.html", tables=table_senti.to_html())


# ------------------ Sentiment for Blurb ------------------
@app.route("/sentiment_blurb", methods=["GET", "POST"])
def sentiment_blurb():
    render_template("senti_b.html")
    if request.method == "POST":
        return redirect(url_for("/sentiment_blurb_result"))
    else:
        return render_template("senti_b.html")


@app.route("/sentiment_blurb_result", methods=["GET", "POST"])
def sentiment_blurb_result():
    user_num = request.form["Number"]

    def bysentiment_blurb(num):
        database.sort_values(by=["Sentiment Blurb"], inplace=True)

        top = database[["Blurb", "Sentiment Blurb"]][-int(num):]
        bottom = database[["Blurb", "Sentiment Blurb"]][:int(num)]

        return top, bottom

    table_senti = bysentiment_blurb(user_num)

    return render_template("search_solution.html", tables=table_senti.to_html())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
