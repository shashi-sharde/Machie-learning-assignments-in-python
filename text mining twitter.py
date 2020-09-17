# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:03:30 2020

@author: shashi
"""


import re # regular expressions 
from wordcloud import WordCloud
from nltk.corpus import stopwords
import tweepy 
from tweepy import OAuthHandler
import matplotlib.pyplot as plt

consumer_key = 'HdaUnjyYoNiCaPug6ZT8m2Y7l'
consumer_secret = '343OSvx7RVPft8QEvRGPIa5La3U05eJX69YUOUpRX0HL8qLECa'
access_token = '1129645192544366592-s8wsd1SC6XvPmbTbGJCRc8gIupTkfI'
access_token_secret = 'TCC32NWZEHs8ZXQ6A9ReT9GiSt3G0ObPEC2q9wDTaeMqq'
  
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
keyword = "realDonaldTrump" 
tweets = api.user_timeline(id=keyword, count=200)

tmp=''  
  
tweets_for_csv = [tweet.text for tweet in tweets] # CSV file created  
for j in tweets_for_csv: 
    # Appending tweets to the empty string tmp 
    tmp+=j  


# Removing unwanted symbols incase if exists
tmp = re.sub("[^A-Za-z" "]+"," ",tmp).lower()
tmp = re.sub("[0-9" "]+"," ",tmp)

# words that contained in  tweets
tweet_words = tmp.split(" ")


with open("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\text_mining\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

tweet_words = [w for w in tweet_words if not w in stopwords]


from afinn import Afinn
af=Afinn()
import pandas as pd
sentiment_scores = [af.score(article) for article in tweet_words]
sentiment_category = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in sentiment_scores]
    

df = pd.DataFrame([ sentiment_scores, sentiment_category]).T
df.columns = [ 'sentiment_score', 'sentiment_category']
df['sentiment_score'] = df.sentiment_score.astype('float')
df.groupby('sentiment_category').describe()

plt.bar(tweet_words,sentiment_scores)
plt.xlabel('Individual Words found in tweets')
plt.ylabel('Sentiment Scores')
plt.show()

# positive words # Choose the path for +ve words stored in system
with open("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\text_mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\text_mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in tweet_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in tweet_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

