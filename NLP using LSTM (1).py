#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[31]:


df=pd.read_csv('/home/gsunilmadhusudanreddy/Training/NLP/fake news-lstm/train.csv')
df_test=pd.read_csv('/home/gsunilmadhusudanreddy/Training/NLP/fake news-lstm/test.csv')


# In[32]:


df.head()


# In[33]:


df_test.head()


# In[34]:


df.info()


# In[35]:


df.drop(['id', 'author'], axis = 1, inplace = True)


# In[36]:


df.dropna(inplace = True)


# In[37]:


df.shape


# In[38]:


y = df['label']
y.value_counts()


# In[39]:


x = df.drop('label', axis = 1)
x.shape


# In[40]:


import tensorflow as tf


# In[41]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


# In[42]:


import nltk
import re
from nltk.corpus import stopwords


# In[43]:


voc_size=5000


# One Hot Representation

# In[44]:


messages=x.copy()


# In[45]:


messages['title'][1]


# In[46]:


messages.reset_index(inplace=True)


# In[48]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[50]:


corpus[18000]


# In[67]:


onehot_repr=[one_hot(words,voc_size)for words in corpus] 


# In[68]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
print(embedded_docs)


# In[69]:


embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# #using dropout
# embedding_vector_features=40
# model1=Sequential()
# model1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
# model1.add(Bidirectional(LSTM(100)))
# model1.add(Dropout(0.3))
# model1.add(Dense(1,activation='sigmoid'))
# model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model1.summary())

# In[70]:


len(embedded_docs),y.shape


# In[71]:


x_final=np.array(embedded_docs)
y_final=np.array(y)


# In[72]:


x_final.shape,y_final.shape


# In[74]:


x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)


# In[76]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=4,batch_size=32)


# In[92]:


predictions = (model.predict(x_test) > 0.5).astype("int32")


# In[93]:


np.unique(predictions)


# In[95]:


confusion_matrix(y_test,predictions)


# In[97]:


accuracy_score(y_test,predictions)


# In[99]:


print(classification_report(y_test,predictions))


# In[ ]:




