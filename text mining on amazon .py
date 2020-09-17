#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[2]:


kindle_reviews =[] 


# In[3]:


for i in range(1,200):
  ip=[]  
  
  url = "https://www.amazon.in/All-New-Kindle-reader-Glare-Free-Touchscreen/product-reviews/B0186FF45G/ref=cm_cr_getr_d_paging_btm_3?showViewpoints=1&pageNumber="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})


# In[4]:


# Extracting the content under specific tags  

for i in range(len(reviews)):
  ip.append(reviews[i].text)  
kindle_reviews=kindle_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews
help(WordCloud)  


# In[5]:


# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(kindle_reviews)


# In[6]:


# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)


# In[7]:


# words that contained in iphone 7 reviews
ip_reviews_words = ip_rev_string.split(" ")


# In[8]:


#stop_words = stopwords.words('english')

with open("Downloads\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


# In[9]:


#stp_wrds = stopwords+stop_words

temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]


ip_reviews_words = [w for w in ip_reviews_words if not w in stopwords]


# In[10]:


# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)


# In[11]:


# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)


# In[13]:


# positive words # Choose the path for +ve words stored in system
with open("Downloads\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]


# In[14]:


# negative words  Choose path for -ve words stored in system
with open("Downloads\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]


# In[15]:


# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)


# In[16]:


# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)

