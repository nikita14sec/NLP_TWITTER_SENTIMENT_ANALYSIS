#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


twitter_df=pd.read_csv('train.csv')


# In[4]:


twitter_df


# In[5]:




twitter_df.info()


# In[6]:


twitter_df.describe()


# In[7]:


twitter_df['tweet']


# In[8]:


twitter_df.drop(['id'],axis=1)


# DATA EXPLORATION
# 

# In[9]:


sns.heatmap(twitter_df.isnull(),yticklabels=False,cbar=False,cmap="Blues")


# In[12]:


twitter_df.hist(bins=30,figsize=(13,5),color='r')


# In[10]:


sns.countplot(twitter_df['label'],label='Count')


# In[11]:


twitter_df['lenght']=twitter_df['tweet'].apply(len)


# In[12]:


twitter_df


# In[13]:


twitter_df.describe()


# In[14]:


twitter_df[twitter_df['lenght']==11]['tweet']


# In[15]:




twitter_df.columns = ['id', 'label', 'tweet','length']


# In[19]:


twitter_df


# In[16]:


twitter_df[twitter_df['length']==84]['tweet'].iloc[0]


# In[21]:


twitter_df['length'].plot(bins=100,kind='hist')


# PLOT THE WORD CLOUD
# 

# In[17]:


positive=twitter_df[twitter_df['label']==0]
positive


# In[18]:


negative=twitter_df[twitter_df['label']==1]
negative


# In[19]:


sentences=twitter_df['tweet'].tolist()
len(sentences)


# In[20]:


sentences_as_one_string=" ".join(sentences)


# In[28]:


sentences_as_one_string


# In[23]:


pip install wordcloud


# In[25]:


from wordcloud import WordCloud


# In[26]:


plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[27]:


sentences_negative=negative['tweet'].tolist()
len(sentences_negative)


# In[28]:


negative_sentences_as_one=" ".join(sentences_negative)


# In[29]:


negative_sentences_as_one


# In[30]:


plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(negative_sentences_as_one))


# PERFORM DATA CLEANING
# 

# In[31]:


import string
string.punctuation


# In[34]:


import nltk
nltk.download()


# In[37]:


from nltk.corpus import stopwords
stopwords.words('english')


# CREATE PIPELINE TO REMOVE PUNCTUATIONS STOPWORDS AND PERFORM COUNT VECTORIZATION
# 

# In[48]:


def message_cleaning(message):
  message_punc_removed=[char for char in message if char not in string.punctuation]
  message_punc_removed_join=''.join(message_punc_removed)
  message_punc_removed_clean=[word for word in message_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
  #message_punc_removed_clean_join=' '.join( message_punc_removed_clean)
  return message_punc_removed_clean


# In[46]:


twitter_df_clean=twitter_df['tweet'].apply(message_cleaning)


# In[47]:


print(twitter_df_clean[5])


# In[49]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer=message_cleaning,dtype=np.uint8)
tweets_countvectorizer=vectorizer.fit_transform(twitter_df['tweet'])


# In[50]:


print(vectorizer.get_feature_names())


# In[51]:


print(tweets_countvectorizer.toarray( ))


# In[52]:


tweets_countvectorizer.shape


# In[53]:


X=pd.DataFrame(tweets_countvectorizer.toarray())


# In[54]:


y=twitter_df['label']


# TRAIN NAIVE BAYES CLASSIFIER MODEL

# In[55]:


X.shape


# In[57]:


y.shape


# In[58]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[59]:


from sklearn.naive_bayes import MultinomialNB


# In[60]:


NB_classifier=MultinomialNB()
NB_classifier.fit(X_train,y_train)


# In[63]:


from sklearn.metrics import classification_report,confusion_matrix


# In[65]:


y_predict_test=NB_classifier.predict(X_test)
cm=confusion_matrix(y_test,y_predict_test)
sns.heatmap(cm,annot=True)


# 

# In[66]:


print(classification_report(y_test,y_predict_test))


# In[ ]:




