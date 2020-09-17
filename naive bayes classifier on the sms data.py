#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn. naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


# In[2]:


data=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on naive base\\sms_raw_NB.csv",encoding='latin-1')
data


# In[15]:



data.info()


# In[16]:


data.describe()


# In[18]:


#doing the eda
from pandas_profiling import ProfileReport
report=ProfileReport(data,title='profile report of the data',explorative=True)
report.to_widgets()


# In[19]:


#it is containing the 7.2% data as duplicte .so removing the the duplicte rows.
data=data.drop_duplicates()
data


# In[20]:


#checking the duplicated rows has been removed or not
a=data.duplicated()
a
#all the duplicate rows has been removed.


# In[22]:


#adding new column to check the length of every messee present in the text clumn
data['length'] = data['text'].apply(len)
data.head()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


data['length'].plot(bins=50,kind='hist')


# In[25]:


data.describe()


# In[39]:


import string

test_message = "Aiya we discuss later lar... Pick u up at 4 is it?"
pre_message = [char for char in test_message if char not in string.punctuation]
pre_message = ''.join(pre_message)
print(pre_message)


# In[40]:


import nltk
nltk.download('stopwords')


# In[57]:


from nltk.corpus import stopwords
stopwords.words('english')[0:30]


# In[58]:


pre_message.split()


# In[59]:


clean_mess = [ word for word in pre_message.split() if word.lower() not in stopwords.words('english')]
clean_mess


# In[60]:


def text_Process(text):
    pre_message = [char for char in text if char not in string.punctuation]
    pre_message = ''.join(pre_message)
    clean_mess = [ word for word in pre_message.split() if word.lower() not in stopwords.words('english')]
    return clean_mess


# In[45]:


data['text'].head(5).apply(text_Process)


# In[61]:


from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_Process).fit(data['text'])
print(len(bow_transformer.vocabulary_))


# In[63]:


df=data['text'][1]
print(df)


# In[64]:


bow = bow_transformer.transform([df])
print(bow)


# In[65]:


bow1 = bow_transformer.transform(data['text'])


# In[66]:


print('Shape of Sparse Matrix: ',bow.shape)
print('Amount of non-zero occurences:',bow.nnz)


# In[67]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(bow1)


# In[69]:


messages_tfidf=tfidf_transformer.transform(bow1)
print(messages_tfidf.shape)


# In[70]:


#model building
model = MultinomialNB().fit(messages_tfidf,data['type'])


# In[71]:


tfidf4 = tfidf_transformer.transform(bow)


# In[72]:


print('predicted:',model.predict(tfidf4)[0])
print('expected:',data.type[3])


# In[73]:


#predicting the type
all_predictions = model.predict(messages_tfidf)
print(all_predictions)


# In[74]:


#spliting the data in to train and test.
from sklearn.model_selection import train_test_split
text_train,text_test,type_train,type_test = train_test_split(data['text'],data['type'],test_size=0.2)


# In[75]:


#creating the pipline which will contain the countvecterizor,classifier,and tfidf transformer.
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
   ( 'bow1',CountVectorizer(analyzer=text_Process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])


# In[76]:


#FITTING THE PIPELINE CREATED  ON THE TRAIN Data.
pipeline.fit(text_train,type_train)


# In[77]:


#printing the classification report  and confusion matrix.
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(data['type'],all_predictions))
print(confusion_matrix(data['type'],all_predictions))


# In[ ]:


#it is giving the accuracy of 98% which is quite good.

