from flask import Flask, render_template, url_for, flash, redirect
from forms import inputData
from pytube import YouTube
from moviepy.editor import *
import requests
import re
from bs4 import BeautifulSoup
import moviepy.editor as mp
from PIL import Image, ImageFilter


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


def analyze():

		#!/usr/bin/env python
	# # Data Cleaning
	import speech_recognition as sr
	from pydub import AudioSegment
	from pydub.silence import split_on_silence
	from pathlib import Path

	# Web scraping, pickle imports
	import pickle
	import time

	def get_large_audio_transcription(path):
	    """
	    Splitting the large audio file into chunks
	    and apply speech recognition on each of these chunks
	    """
	    total_chunk = 0
	    error_cnt = 0

	    #init recognizer
	    r = sr.Recognizer()
	    # open the audio file using pydub
	    sound = AudioSegment.from_wav(path)
	    # split audio sound where silence is 700 miliseconds or more and get chunks
	    chunks = split_on_silence(sound,
	                              # experiment with this value for your target audio file
	                              min_silence_len=300,
	                              # adjust this per requirement
	                              silence_thresh=sound.dBFS - 16,
	                              # keep the silence for 1 second, adjustable as well
	                              keep_silence=500,
	                              )
	    folder_name = "  audio-chunks"
	    # create a directory to store the audio chunks
	    if not os.path.isdir(folder_name):
	        os.mkdir(folder_name)
	    whole_text = ""
	    # process each chunk
	    for i, audio_chunk in enumerate(chunks, start=1):
	        # export audio chunk and save it in
	        # the `folder_name` directory.
	        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
	        audio_chunk.export(chunk_filename, format="wav")
	        # recognize the chunk
	        with sr.AudioFile(chunk_filename) as source:
	            audio_listened = r.record(source)
	            # try converting it to text
	            try:
	                text = r.recognize_google(audio_listened)
	                total_chunk += 1
	            except sr.UnknownValueError as e:
	                error_cnt += 1
	                print("Error:", str(e))
	            else:
	                text = f"{text.capitalize()}. "
	                print(chunk_filename, ":", text)
	                whole_text += text
	    # return the text for all chunks detected
	    print(f"total chunk is {total_chunk}, error chunk is {error_cnt}")
	    return whole_text


	from mutagen.mp3 import MP3
	def audio2txt(fileName):
	    mp3path = Path("./" + fileName + ".mp3")
	    wavpath = Path("./" + fileName + ".wav")

	    print(mp3path)
	    source = AudioSegment.from_mp3(mp3path)
	    source.export(wavpath, format="wav")

	    # checking how long is the audio, so we can get words per minutes later on
	    audio = MP3(mp3path)
	    # print("audio length is " + str(audio.info.length) + " seconds")  # unit is seconds
	    duration = audio.info.length / 60.0

	    # a function that splits the audio file into chunks
	    # and applies speech recognition
	    result = get_large_audio_transcription(wavpath)
	    # print(result)
	    # this is rather slow process, should save the result

	    textfile = Path("./transcripts" + fileName + ".txt")

	    # first time run, let's save this into text
	    save_text = open(textfile, 'w', encoding='utf-8')
	    save_text.write(result)
	    save_text.close()

	    # just read transcription from text
	    # load_text = open(textfile, 'r', encoding='utf-8')
	    # transcription = load_text.read()
	    # load_text.close()
	    # print(transcription)
	    return result, duration

	# todo: put these transcription and duration in a for loop, to automatically append itself

	speaker1 = "Comparison"
	speaker2 = "Myself"
	start_time = time.time()
	transcription0, time0 = audio2txt(speaker1)
	print(f"it took {time.time() - start_time} to convert first audio")
	start_time = time.time()
	transcription1, time1 = audio2txt(speaker2)
	print(f"it took {time.time() - start_time} to convert second audio")
	print(transcription0)
	print("the first speech duration is " + str(time0) + " minutes\n")
	print(transcription1)
	print("the second speech duration is " + str(time1) + " minutes\n")

	transcription = [transcription0, transcription1] # string in list
	duration = [time0, time1]

	# count average syllable per word
	# unfortunately the rule is complicated, there are too many exceptions
	# check out https://eayd.in/?p=232
	def sylco(word):
	    word = word.lower()

	    # exception_add are words that need extra syllables
	    # exception_del are words that need less syllables

	    exception_add = ['serious', 'crucial']
	    exception_del = ['fortunately', 'unfortunately']

	    co_one = ['cool', 'coach', 'coat', 'coal', 'count', 'coin', 'coarse', 'coup', 'coif', 'cook', 'coign', 'coiffe',
	              'coof', 'court']
	    co_two = ['coapt', 'coed', 'coinci']

	    pre_one = ['preach']

	    syls = 0  # added syllable number
	    disc = 0  # discarded syllable number

	    # 1) if letters < 3 : return 1
	    if len(word) <= 3:
	        syls = 1
	        return syls

	    # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
	    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

	    if word[-2:] == "es" or word[-2:] == "ed":
	        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
	        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
	            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[
	                                                                                                       -3:] == "ies":
	                pass
	            else:
	                disc += 1

	    # 3) discard trailing "e", except where ending is "le"

	    le_except = ['whole', 'mobile', 'pole', 'male', 'female', 'hale', 'pale', 'tale', 'sale', 'aisle', 'whale', 'while']

	    if word[-1:] == "e":
	        if word[-2:] == "le" and word not in le_except:
	            pass

	        else:
	            disc += 1

	    # 4) check if consecutive vowels exists, triplets or pairs, count them as one.

	    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
	    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
	    disc += doubleAndtripple + tripple

	    # 5) count remaining vowels in word.
	    numVowels = len(re.findall(r'[eaoui]', word))

	    # 6) add one if starts with "mc"
	    if word[:2] == "mc":
	        syls += 1

	    # 7) add one if ends with "y" but is not surrouned by vowel
	    if word[-1:] == "y" and word[-2] not in "aeoui":
	        syls += 1

	    # 8) add one if "y" is surrounded by non-vowels and is not in the last word.

	    for i, j in enumerate(word):
	        if j == "y":
	            if (i != 0) and (i != len(word) - 1):
	                if word[i - 1] not in "aeoui" and word[i + 1] not in "aeoui":
	                    syls += 1

	    # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

	    if word[:3] == "tri" and word[3] in "aeoui":
	        syls += 1

	    if word[:2] == "bi" and word[2] in "aeoui":
	        syls += 1

	    # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

	    if word[-3:] == "ian":
	        # and (word[-4:] != "cian" or word[-4:] != "tian") :
	        if word[-4:] == "cian" or word[-4:] == "tian":
	            pass
	        else:
	            syls += 1

	    # 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

	    if word[:2] == "co" and word[2] in 'eaoui':

	        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
	            syls += 1
	        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one:
	            pass
	        else:
	            syls += 1

	    # 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

	    if word[:3] == "pre" and word[3] in 'eaoui':
	        if word[:6] in pre_one:
	            pass
	        else:
	            syls += 1

	    # 13) check for "-n't" and cross match with dictionary to add syllable.

	    negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]

	    if word[-3:] == "n't":
	        if word in negative:
	            syls += 1
	        else:
	            pass

	            # 14) Handling the exceptional words.

	    if word in exception_del:
	        disc += 1

	    if word in exception_add:
	        syls += 1

	        # calculate the output
	    return numVowels - disc + syls

	# # count average words per sentence
	import re
	idx = 0
	total_syllable = 0
	for trans in transcription:
	    parts = [len(l.split()) for l in re.split(r'[?!.]', trans) if l.strip()]
	    # print(parts) # sanity check

	    if idx == 0:
	        print(f"for {speaker1}: ")
	        # print average words per minute
	        print(f"average words per minute: {(sum(parts) / duration[idx]):0.1f}")
	        print(f"total speech duration is {duration[idx]:0.1f} minutes")

	    else:
	        print(f"for {speaker2}: ")
	        # print average words per minute
	        print(f"average words per minute : {(sum(parts) / duration[idx]):0.1f}")
	        print(f"total speech duration is {duration[idx]:0.1f} minutes")
	        # for word in trans:
	        total_syllable = sylco(trans)
	        # ASW is average number of syllables per word
	        ASW = total_syllable / sum(parts)
	        # flesch-kincaid grade level test 0.39 * ASL + 11.8 * ASW - 15.59
	        grade_level = 0.39 * ASL + 11.8 * ASW - 15.5019
	        print(f"average syllables per word is {ASW:0.2f}")
	        # print(f"Flesch-Kincaid Grade Level test result is {grade_level:0.2f}")
	    idx += 1
	    # ASL = average sentence length
	    # (the number of words divided by the number of sentences)
	    ASL = sum(parts)/(len(parts)+1)
	    print(f"average words per sentence is : {ASL:0.2f}")
	    total_syllable += sylco(trans)
	    print(f"total syllable is {total_syllable:0.2f}")
	    # ASW is average number of syllables per word
	    ASW = total_syllable / (sum(parts)+1)
	    # flesch-kincaid grade level test 0.39 * ASL + 11.8 * ASW - 15.59
	    grade_level = 0.39 * ASL + 11.8 * ASW - 15.59
	    print(f"average syllables per word is {ASW:0.2f}")
	    # print(f"Flesch-Kincaid Grade Level test result is {grade_level}")

	    total_syllable = 0

	def url_to_transcript(url):
	    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
	    page = requests.get(url).text
	    soup = BeautifulSoup(page, "lxml")
	    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
	    print(url)
	    return text


	def url_to_transcript_entry_content(url):
	    '''Returns transcript data specifically from scrapsfromtheloft.com.'''
	    page = requests.get(url).text
	    soup = BeautifulSoup(page, "lxml")
	    text = [p.text for p in soup.find(class_="entry-content").find_all('p')]
	    print(url)
	    return text


	# page = requests.get('https://quest4curiosity.com/2020/06/18/data-science-allergies/').text
	# soup = BeautifulSoup(page, "lxml")
	# text = [p.text for p in soup.find(class_="entry-content").find_all('p')]


	# tianyi_urls = ['https://quest4curiosity.com/2020/09/26/how-justice-ruth-bader-ginsburg-became-known-as-the-notorious-rbg/',
	#                'https://quest4curiosity.com/2010/02/13/example-post/',
	#                'https://quest4curiosity.com/2020/08/16/time-for-change-in-the-climate-or-how-we-act/']
	# tianyi_transcripts = [url_to_transcript_entry_content(u) for u in tianyi_urls]
	transcription = [[transcription[0]], [transcription[1]]] # list in list for later process


	tianyi = [speaker1, speaker2]

	# Pickle files for later use

	# Make a new directory to hold the text files
	# !mkdir transcripts

	for i, c in enumerate(tianyi):
	    with open("transcripts" + c + ".txt", "wb") as file:
	        pickle.dump(transcription[i], file)

	# In[88]:


	# Load pickled files

	data_tianyi = {}
	for i, c in enumerate(tianyi):
	    with open("transcripts" + c + ".txt", "rb") as file:
	        data_tianyi[c] = pickle.load(file)
	print("check pickled field\n")
	print(data_tianyi.keys())


	# Double check to make sure data has been loaded properly
	data_tianyi.keys()

	# ## Cleaning The Data

	# Let's take a look at our data again
	# next(iter(data.keys()))
	next(iter(data_tianyi.keys()))

	# Notice that our dictionary is currently in key: comedian, value: list of text format
	# next(iter(data.values()))
	next(iter(data_tianyi.values()))


	# We are going to change this to key: comedian, value: string format
	def combine_text(list_of_text):
	    '''Takes a list of text and combines them into one large chunk of text.'''
	    combined_text = ' '.join(list_of_text)
	    return combined_text

	# Combine it!
	data_tianyi_combined = {key: [combine_text(value)] for (key, value) in data_tianyi.items()}

	# We can either keep it in dictionary format or put it into a pandas dataframe
	import pandas as pd

	pd.set_option('max_colwidth', 150)

	data_tianyi_df = pd.DataFrame.from_dict(data_tianyi_combined).transpose()
	data_tianyi_df.columns = ['transcript']
	data_tianyi_df = data_tianyi_df.sort_index()
	print("data_tianyi_df data:\n")
	print(type(data_tianyi_df))

	# Let's take a look at the transcript for Ali Wong
	# # data_df.transcript.loc['ali']

	# Apply a first round of text cleaning techniques
	import re
	import string


	def clean_text_round1(text):
	    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
	    text = text.lower()
	    text = re.sub('\[.*?\]', '', text)
	    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	    text = re.sub('\w*\d\w*', '', text)
	    return text


	round1 = lambda x: clean_text_round1(x)

	# Let's take a look at the updated text
	data_tianyi_clean = pd.DataFrame(data_tianyi_df.transcript.apply(round1))
	print(data_tianyi_clean)


	# Apply a second round of cleaning
	def clean_text_round2(text):
	    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
	    text = re.sub('[‘’“”…]', '', text)
	    text = re.sub('\n', '', text)
	    return text


	round2 = lambda x: clean_text_round2(x)

	# Let's take a look at the updated text
	data_tianyi_clean = pd.DataFrame(data_tianyi_clean.transcript.apply(round2))
	print(data_tianyi_clean)

	# ## Organizing The Data

	# Let's take a look at our dataframe
	data_tianyi_df

	# Let's add the comedians' full names as well
	tianyi_topics = [speaker2, speaker1]
	data_tianyi_df['topic'] = tianyi_topics
	data_tianyi_df


	# Let's pickle it for later use
	data_tianyi_df.to_pickle("./corpus_tianyi.pkl")

	# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
	from sklearn.feature_extraction.text import CountVectorizer

	cv = CountVectorizer(stop_words=None)
	data_tianyi_cv = cv.fit_transform(data_tianyi_clean.transcript)
	data_tianyi_dtm = pd.DataFrame(data_tianyi_cv.toarray(), columns=cv.get_feature_names())
	data_tianyi_dtm.index = data_tianyi_clean.index
	data_tianyi_dtm

	# # Let's pickle it for later use
	data_tianyi_dtm.to_pickle("./dtm_tianyi.pkl")


	# Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
	data_tianyi_clean.to_pickle('./data_tianyi_clean.pkl')
	pickle.dump(cv, open("./cv_tianyi.pkl", "wb"))

	# # Exploratory Data Analysis

	# Read in the document-term matrix
	import pandas as pd

	data_tianyi = pd.read_pickle('./dtm_tianyi.pkl')
	data_tianyi = data_tianyi.transpose()
	data_tianyi.head()

	# Find the top 30 words said by each comedian
	top_dict = {}
	for c in data_tianyi.columns:
	    top = data_tianyi[c].sort_values(ascending=False).head(30)
	    top_dict[c] = list(zip(top.index, top.values))

	top_dict

	# Print the top 15 words said by each comedian
	for text, top_words in top_dict.items():
	    print(text)
	    print(', '.join([word for word, count in top_words[0:14]]))
	    print('---')

	# Look at the most common top words --> add them to the stop word list
	from collections import Counter

	# Let's first pull out the top 30 words for each person
	words = []
	for text in data_tianyi.columns:
	    top = [word for (word, count) in top_dict[text]]
	    for t in top:
	        words.append(t)


	# Let's aggregate this list and identify the most common words along with how many routines they occur in
	Counter(words).most_common()


	# If more than half of the comedians have it as a top word, exclude it from the list
	add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
	add_stop_words


	# Let's update our document-term matrix with the new list of stop words
	from sklearn.feature_extraction import text
	from sklearn.feature_extraction.text import CountVectorizer

	# Read in cleaned data
	data_tianyi_clean = pd.read_pickle('./data_tianyi_clean.pkl')

	# Add new stop words
	stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

	# Recreate document-term matrix
	cv = CountVectorizer(stop_words=stop_words)
	data_tianyi_cv = cv.fit_transform(data_tianyi_clean.transcript)
	data_tianyi_stop = pd.DataFrame(data_tianyi_cv.toarray(), columns=cv.get_feature_names())
	data_tianyi_stop.index = data_tianyi_clean.index

	# Pickle it for later use
	import pickle

	pickle.dump(cv, open("/cv_tianyi_stop.pkl", "wb"))
	data_tianyi_stop.to_pickle("dtm_tianyi_stop.pkl")


	# Let's make some word clouds!
	# Terminal / Anaconda Prompt: conda install -c conda-forge wordcloud
	from wordcloud import WordCloud

	wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
	               max_font_size=150, random_state=42)

	# Reset the output dimensions
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec

	fig1, f1_axes = plt.subplots(ncols=3, nrows=2, constrained_layout=True)

	plt.rcParams['figure.figsize'] = [16, 6]

	tianyi_topics = [speaker2, speaker1]
	# Create subplots for each comedian
	for index, text in enumerate(data_tianyi.columns):
	    wc.generate(data_tianyi_clean.transcript[text])

	    plt.subplot(2, 1, index + 1)
	    plt.imshow(wc, interpolation="bilinear")
	    plt.axis("off")
	    plt.title(tianyi_topics[index])
	plt.savefig("./static/word.png")
	# plt.show()


	# Find the number of unique words that each text uses

	# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
	unique_list = []
	for text in data_tianyi.columns:
	    uniques = data_tianyi[text].to_numpy().nonzero()[0].size
	    unique_list.append(uniques)

	# Create a new dataframe that contains this unique word count
	data_tianyi_words = pd.DataFrame(list(zip(tianyi_topics, unique_list)), columns=['text', 'unique_words'])
	data_tianyi_unique_sort = data_tianyi_words.sort_values(by='unique_words')
	data_tianyi_unique_sort


	# Calculate the words per minute of each comedian

	# Find the total number of words that a comedian uses
	total_list = []
	for text in data_tianyi.columns:
	    totals = sum(data_tianyi[text])
	    total_list.append(totals)

	# Comedy special run times from IMDB, in minutes
	run_times = [time0, time1]

	# Let's add some columns to our dataframe
	data_tianyi_words['total_words'] = total_list
	data_tianyi_words['run_times'] = run_times
	data_tianyi_words['words_per_minute'] = data_tianyi_words['total_words'] / data_tianyi_words['run_times']

	# Sort the dataframe by words per minute to see who talks the slowest and fastest
	data_wpm_sort = data_tianyi_words.sort_values(by='words_per_minute')
	data_wpm_sort


	# Let's plot our findings
	import numpy as np

	y_pos = np.arange(len(data_tianyi_words))

	plt.subplot(1, 2, 1)
	plt.barh(y_pos, data_tianyi_unique_sort.unique_words, align='center')
	plt.yticks(y_pos, data_tianyi_unique_sort.text)
	plt.title('Number of Unique Words', fontsize=20)

	# plt.subplot(1, 2, 2)
	# plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
	# plt.yticks(y_pos, data_wpm_sort.comedian)
	# plt.title('Number of Words Per Minute', fontsize=20)

	plt.tight_layout()
	plt.savefig("./static/variety.png")
	# plt.show()

	plt.clf()
	# We'll start by reading in the corpus, which preserves word order
	import pandas as pd

	data_tianyi = pd.read_pickle('./corpus_tianyi.pkl')
	data_tianyi


	# Create quick lambda functions to find the polarity and subjectivity of each routine
	# Terminal / Anaconda Navigator: conda install -c conda-forge textblob
	from textblob import TextBlob

	pol = lambda x: TextBlob(x).sentiment.polarity
	sub = lambda x: TextBlob(x).sentiment.subjectivity

	data_tianyi['polarity'] = data_tianyi['transcript'].apply(pol)
	data_tianyi['subjectivity'] = data_tianyi['transcript'].apply(sub)
	data_tianyi


	# Let's plot the results
	import matplotlib.pyplot as plt

	plt.rcParams['figure.figsize'] = [10, 8]

	for index, text in enumerate(data_tianyi.index):
	    x = data_tianyi.polarity.loc[text]
	    y = data_tianyi.subjectivity.loc[text]
	    plt.scatter(x, y, color='blue')
	    plt.text(x + .001, y + .001, data_tianyi['topic'][index], fontsize=10)
	    plt.xlim(-1, 1)

	plt.title('Sentiment Analysis', fontsize=20)
	plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
	plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)
	plt.savefig("./static/sentimentpolarity.png")
	# plt.show()

def cleanup():
	os.remove('Comparison.mp3')
	os.remove('Comparison.wav')
	os.remove('myself.mp3')
	os.remove('myself.wav')





@app.route("/")
@app.route("/try")
@app.route("/home")
@app.route("/explore")
def home():
    return render_template('home.html')

@app.route("/about")
@app.route("/learn")
def about():
    return render_template('about.html')


@app.route("/results")
def results():
	analyze()
	cleanup()
	return render_template('results.html')

from pytube import YouTube
from moviepy.editor import *
import requests	
import re
from bs4 import BeautifulSoup
import moviepy.editor as mp
import os

def pytube(link, num):
	
	# https://simply-python.com/2019/01/02/downloading-youtube-videos-and-converting-to-mp3/
	# DOWNLOADING YOUTUBE VIDEOS AND CONVERTING TO MP3
	# A simple guide to download videos from YouTube using
	# Required
	# Tools:
	# PyTube— primarily for downloading youtube videos.
	# MoviePy — for video editing and also convert to mp3.
	# Steps:
	# pip install pytube and moviepy
	# Basic Usage
	# download a file from youtube
	# youtube_link = 'https://www.youtube.com/watch?v=yourtubevideos'
	# youtube_link = 'https://www.youtube.com/watch?v=xH_B6xh42xc&list=PLPKaZQAWre5bWqn0Fmm8v7XjIOcPyXeEz&index=3'
	# w = YouTube(youtube_link).streams.first()
	# w.download(output_path="./youtube")
	# download a file with only audio, to save space
	# if the final goal is to convert to mp3
	# youtube_link = 'https://www.youtube.com/watch?v=targetyoutubevideos'
	# https://www.youtube.com/watch?v=u6jglW-vCAo
	# https://www.youtube.com/watch?v=s0WEFTZS3SQ
	placeholder = str(link)
	if num==0:
		placeholder = placeholder[67:]

		# placeholder = placeholder.translate({ord(i): None for i in '<input id="firstlink" name="firstlink" required type="text" value="'})
		# placeholder = placeholder.translate({ord(i): None for i in '">'})
		print(placeholder)
	else:
		placeholder = placeholder[69:]
		# placeholder = placeholder.translate({ord(i): None for i in '<input id ="secondlink" name="secondlink" required type="text" value="'})
		# placeholder = placeholder.translate({ord(i): None for i in '">'})
		print(placeholder)
	youtube_link = placeholder

	y = YouTube(youtube_link)
	t = y.streams.filter(only_audio=True).all()
	t[0].download(output_path="./")
	# Downloading videos from a YouTube playlist

	# website = 'https://www.youtube.com/playlist?list=yourfavouriteplaylist'
	# website = 'https://www.youtube.com/watch?v=s0WEFTZS3SQ'
	r = requests.get(youtube_link)
	soup = BeautifulSoup(r.text)

	tgt_list = [a['href'] for a in soup.find_all('a', href=True)]
	# print(tgt_list)
	tgt_list = [n for n in tgt_list if re.search('watch', n)]


	unique_list = []
	for n in tgt_list:
	    if n not in unique_list:
	        unique_list.append(n)

	# all the videos link in a playlist
	unique_list = ['https://www.youtube.com' + n for n in unique_list]

	for link in unique_list:
	    print(link)
	    y = YouTube(link)
	    t = y.streams.all()
	    t[0].download(output_path="./")

	# Converting from MP4 to MP3 (from a folder with mp4 files)
	tgt_folder = "./"

	for file in [n for n in os.listdir(tgt_folder) if re.search('mp4', n)]:
	    full_path = os.path.join(tgt_folder, file)
	output_path = os.path.join(tgt_folder, os.path.splitext(file)[0] + '.mp3')
	clip = mp.AudioFileClip(full_path).subclip(10, )  # disable if do not want any clipping
	clip.write_audiofile(output_path)

	print(output_path)
	print(type(output_path))
	title = output_path[2:]
	mp4title=title.rstrip(title[-1])+'4'
	if num==0:
		os.rename(title, 'Comparison.mp3')
	else:
		os.rename(title, 'myself.mp3')
	# 	os.rename(r'C:\Users\Tianyi Gu\Flaskapp\youtube\_' + str(output_path) + '.mp3', r'C:\Users\Tianyi Gu\Flaskapp\youtube\myself.mp3')
	os.remove(mp4title)






@app.route("/input", methods=['GET', 'POST'])
def input():
	form = inputData()
	result = {}
	if form.validate_on_submit():
		flash(f'Comparisons received!')
		results=[form.firstlink, form.secondlink]
		# print(results[0])
		# print(results[1])
		pytube(results[0],0)
		pytube(results[1],1)
		return redirect(url_for('results'))
	return render_template('input.html', form=form, result=result)



if __name__ == '__main__':
    app.run(debug=True)
