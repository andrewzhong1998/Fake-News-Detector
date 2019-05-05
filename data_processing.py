#!/usr/bin/env python
# coding: utf-8

# <b>Import data<b>

# In[1]:


import pandas

articles1 = pandas.read_csv("/home/ziwei75/Desktop/final_project/articles1.csv")
fakearticle = pandas.read_csv("/home/ziwei75/Desktop/final_project/fake.csv")


# <b>Process article <b>

# In[2]:


import numpy as np


# In[3]:


articles1 = articles1[:10000]
fakearticle = fakearticle[:10000]
articles1_content = articles1.drop(["Unnamed: 0", "id", "publication","author","date","year","month","url"], axis = 1)
articles1_array = np.asarray(articles1_content)
fakearticle_content = fakearticle.drop(['uuid', 'ord_in_thread','published','author','language','crawled','site_url','country','domain_rank','thread_title','spam_score','main_img_url','replies_count','participants_count','likes','comments','shares','type'], axis = 1)
fake_array = np.asarray(fakearticle_content)


# <b> create dictionary <b>

# In[4]:


dict = {}
count = 0
list = []


# In[5]:


import re
datas = [articles1_array, fake_array]
for data in datas:
    N = np.shape(data)[0];
    for i in range(N):
        if((type(data[i,0]) is not str)):
            continue;
        if((type(data[i,1]) is not str)):
            continue;
        title = re.sub(r'[^\w\s]','',data[i,0]).upper().split()
        content = re.sub(r'[^\w\s]','',data[i,1]).upper().split()
        content.extend(title)
        for word in content:
            if(dict.get(word) is None):
                dict[word]=count;
                count+=1;


# In[6]:


print(len([*dict]))


# In[7]:


dictCopy = dict


# In[8]:


vocab_index = []
for vocab, index in dictCopy.items():
    vocab_index += [(vocab, index)]


# In[9]:


with open("word_index_mapping","w") as f:
    for vocab, index in vocab_index:
        f.write("%s, %s \n"  %(vocab, index))


# In[10]:


word_index_mapping = open("word_index_mapping").readlines()


# In[11]:


word_index_mapping[:10]


# <b>Create Feature<b>

# In[14]:


N1 = np.shape(articles1_array)[0]
N4 = np.shape(fake_array)[0]
dataset_length = N1 + N4
F = len([*dict])
X = np.zeros((dataset_length, F))
datas = [articles1_array, fake_array]
count = 0;
for data in datas:
    N = np.shape(data)[0];
    for i in range(N):
        if((type(data[i,0]) is not str)):
            continue;
        if((type(data[i,1]) is not str)):
            continue;
        title = re.sub(r'[^\w\s]','',data[i,0]).upper().split()
        content = re.sub(r'[^\w\s]','',data[i,1]).upper().split()
        content.extend(title)
        for word in content:
            X[count, dict[word]] += 1
        count+=1
        #print(count)
    print("finished")


# #### shuffle and split dataset

# In[15]:


features = X
del X


# In[16]:


fake_length = N4
lab = [1]*(dataset_length-fake_length)
neg = [0]*fake_length
lab.extend(neg)


# In[17]:


index = np.arange(0,dataset_length)


# In[18]:


np.random.shuffle(index)


# In[19]:


np.save("article_index.npy",index)


# In[20]:


training_index1 = index[:int(dataset_length*0.4)]
training_features1 = np.take(features, training_index1, axis=0)
training_labels1 = np.take(lab, training_index1)


# In[21]:


np.save("training_features1.npy",training_features1)
np.save("training_labels1.npy",training_labels1)


# In[22]:


np.save("training_labels1.npy",training_labels1)


# In[23]:


del training_features1 
del training_labels1
del training_index1


# In[24]:


training_index2 = index[int(dataset_length*0.4):int(dataset_length*0.8)]
training_features2 = np.take(features, training_index2, axis=0)
training_labels2 = np.take(lab, training_index2)


# In[25]:


np.save("training_features2.npy",training_features2)
np.save("training_labels2.npy",training_labels2)


# In[26]:


del training_features2
del training_labels2


# In[27]:


del training_index2


# In[28]:


test_index = index[int(dataset_length*0.8):int(dataset_length*0.9)]
test_features = np.take(features, test_index, axis=0)
test_labels = np.take(lab, test_index)


# In[29]:


np.save("test_features.npy",test_features)
np.save("test_labels.npy",test_labels)


# In[30]:


del test_features
del test_labels
del test_index


# In[31]:


validation_index = index[int(dataset_length*0.9):]
validation_features = np.take(features, validation_index, axis=0)
validation_labels = np.take(lab, validation_index)


# In[32]:


np.save("validation_features.npy", validation_features)
np.save("validation_labels.npy", validation_labels)


# In[33]:


validation_features.shape


# In[35]:


import numpy as np
training = np.load("./training_features1.npy")


# In[36]:


print(training.shape)


# In[38]:


np.load("./article_index.npy")


# In[40]:


training_index1 = index[:int(dataset_length*0.4)]
training_index2 = index[int(dataset_length*0.4):int(dataset_length*0.8)]
test_index = index[int(dataset_length*0.8):int(dataset_length*0.9)]
validation_index = index[int(dataset_length*0.9):]


# In[45]:


np.save("./index/training_index1",training_index1)


# In[46]:


np.save("./index/training_index2",training_index2)


# In[47]:


np.save("./index/test_index",test_index)


# In[48]:


np.save("./index/validation_index",validation_index)


# In[55]:


total_content = []
for content in articles1_content.values:
    total_content += [content]


# In[59]:


for content in fakearticle_content.values:
    total_content += [content]


# In[75]:


import csv
with open('total_contents.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["title","content"])
    for title, article in total_content:
        writer.writerow([title, article])


# In[76]:


import pandas as pd
pd.read_csv("total_contents.csv")


# In[ ]:




