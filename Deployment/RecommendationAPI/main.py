#!/usr/bin/env python
# coding: utf-8

# In[11]:

#Load the Libraries
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from nltk import cluster
from scipy import spatial
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import torch
import random
torch.use_deterministic_algorithms(False)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import re
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Required downloads for use with above models
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')


# In[13]:


embeddingDim = 300
re_wordMatcher = re.compile(r'[a-z0-9]+') 
maxlen=4

#Function to get FastText Model,Vocabulary and FastText Pretrained Weights
def getWord2Vec(embeddingDim):
    #en_model = KeyedVectors.load_word2vec_format('wiki.multi.en.vec')
    model = pickle.load(open('model.pkl','rb'))
    vocab = list(model.index_to_key)
    print("Vocab size in pretrained model:", len(vocab))

    # check if the word 'and' is present in the pretrained model
    assert "and" in model

    # check the dimension of the word vectors
    assert embeddingDim == len(model["and"])

    # initialize a numpy matrix which will store the word vectors
    # first row is for the padding token
    pretrained_weights = np.zeros((1+len(vocab), embeddingDim))

    # tqdm just adds a progress bar
    for i, token in enumerate(vocab):
        pretrained_weights[i, :] = model[token]

    # map tokens in the vocab to ids
    vocab = dict(zip(vocab, range(1, len(vocab)+1)))
    return model,vocab,pretrained_weights
    


# In[14]:

#Function to get Features from Text(Word Embeddings)
def reviewText2Features(reviewText,en_model,vocab):
    """
    Function which takes review text (basically a string!) as input and returns a features matrix X of shape
    (maxlen, embeddingDim). This is done by splitting the review into words and then representing each word by it's
    word vector obtained from the Word2Vec model. Sentences having more than maxlen words are truncated while shorter
    ones are zero-padded by pre-adding all zero vectors.
    """
    X = []
    
    reviewWords = re_wordMatcher.findall(reviewText.lower())
    

    for i, word in enumerate(reviewWords):
        if word not in en_model:
            continue
        if i >= maxlen:
            break
        # X.append(en_model[word])
        X.append(vocab[word])
    

    if len(X) < maxlen:
        # zero_padding = [[0.]*embeddingDim]*(maxlen - len(X))
        zero_padding = [0.]*(maxlen - len(X))
        X = zero_padding + X
    
    return X # np.array(X)


# In[15]:

#Model Class
class SentimentNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_weights):
        super(SentimentNet, self).__init__()
        
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
        
        """
        Adding a dropout layer to force some of the feature values to zero.
        Note: Dropout is a regularization technique which sets the activation of few randomly chosen neurons of
        a hidden layer to zero. It can also be applied to the input layer where some of the input features are set to zero.
        For more details refer http://jmlr.org/papers/v15/srivastava14a.html
        """
        self.sentInputDropout = nn.Dropout(0.2)
        
        """
        Now let's stack a couple of bidirectional RNNs to process the input sequence and extract features
        """
        self.biLSTM1 = nn.LSTM(embedding_dim, hidden_dim[0], bidirectional=True, batch_first=True)
        self.biLSTMDropOut = nn.Dropout(0.2)
        self.dense1 = nn.Linear(2*hidden_dim[0], 128)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.2)

        self.dense2 = nn.Linear(128, 64)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2)

        self.dense3 = nn.Linear(64, 32)
        self.tanh3 = nn.Tanh()
        self.dropout3 = nn.Dropout(0.2)


        self.outputLayer = nn.Linear(32, 3)
        #self.softmax = nn.Softmax()

        
    def forward(self, x):
        
        batch_len = x.shape[0]
        out = self.embedding(x)
        out = self.sentInputDropout(out)
        out, hidden = self.biLSTM1(out)
        out = self.biLSTMDropOut(out)

        out = self.dense1(out)
        out = self.tanh1(out)
        out = self.dropout1(out)

        out = self.dense2(out)
        out = self.tanh2(out)
        out = self.dropout2(out)

        out = self.dense3(out)
        out = self.tanh3(out)
        out = self.dropout3(out)

        out = self.outputLayer(out)
        #out = self.softmax(out)
        out = out[:,-1]
        return out   


# In[16]:

#Load the Saved Model. This is same model that is used to get Profile_Skill Matrix.
def getModel(embeddingDim,vocab,pretrained_weights):
    model = SentimentNet(embeddingDim, [256], 1+len(vocab), pretrained_weights)
    model.load_state_dict(torch.load('state_dict10.pt',map_location=torch.device('cpu')))
    return model
    


# In[17]:

#Function to get probabilities for each phrase from the saved model
def predictSentiment(reviewText,model,en_model,vocab):
    X = reviewText2Features(reviewText,en_model,vocab)
    X = np.array(X).reshape((1, -1))
    X = torch.from_numpy(X)
    model.eval()
    pred_proba = model(X.long())
    pred_proba = pred_proba.cpu().squeeze().detach().numpy()
    return pred_proba


# In[18]:

#Function to clean text
def clean_text(raw):
    '''Case specific to be used with pandas apply method'''
    try:
        # remove carriage returns and new lines
        raw = raw.replace('\r', '')
        raw = raw.replace('\n', '')
        
        # brackets appear in all instances
        raw = raw.replace('[', '')
        raw = raw.replace(']', '')
        raw = raw.replace(')', '')
        raw = raw.replace('(', '')
        
        # removing html tags
        clean_html = re.compile('<.*?>')
        clean_text = re.sub(clean_html, ' ', raw)
        
        # removing duplicate whitespace in between words
        clean_text = re.sub(" +", " ", clean_text) 
        
        # stripping first and last white space 
        clean_text = clean_text.strip()
        
        # commas had multiple spaces before and after in each instance
        clean_text = re.sub(" , ", ", ", clean_text) 
        
        # eliminating the extra comma after a period
        clean_text = clean_text.replace('.,', '.')
        
        clean_text = clean_text.replace('_x000D_', ' ')
        clean_text = clean_text.replace("'", '')
        
        clean_text = clean_text.encode('ascii','ignore')
        clean_text = clean_text.decode('ascii','ignore')
        

        
        # using try and except due to Nan in the column
    except:
        clean_text = np.nan
        
    return clean_text


# In[19]:

#Function to get POS tag for PWDs Bio
def pos_series(keyword):
    '''categorizes after tokenizing words with POS tags'''
    tokens = nltk.word_tokenize(keyword)
    tagged = nltk.pos_tag(tokens)
    return tagged


# In[20]:

#Function to get Phrases from POS Tagged data
def grammar(pos_tagged_arrs):
    grammar1 = ('''Noun Phrases: {<DT>?<JJ>*<NN|NNS|NNP>+}''')
    chunkParser = nltk.RegexpParser(grammar1)
    tree1 = chunkParser.parse(pos_tagged_arrs)
    grammar2 = ('''NP2: {<IN>?<JJ|NN>*<NNS|NN>}''')
    chunkParser = nltk.RegexpParser(grammar2)
    tree2 = chunkParser.parse(pos_tagged_arrs)
    grammar3 = ('''VS: {<VBG|VBZ|VBP|VBD|VB|VBN><NNS|NN>*}''')
    chunkParser = nltk.RegexpParser(grammar3)
    tree3 = chunkParser.parse(pos_tagged_arrs)
    grammar4 = ('''
    Commas: {<NN|NNS>*<,><NN|NNS>*<,><NN|NNS>*} 
    ''')
    chunkParser = nltk.RegexpParser(grammar4)
    tree4 = chunkParser.parse(pos_tagged_arrs)

    g1_chunks = []
    for subtree in tree1.subtrees(filter=lambda t: t.label() == 'Noun Phrases'):
        g1_chunks.append(subtree)
    g2_chunks = []
    for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP2'):
        g2_chunks.append(subtree)
    g3_chunks = []
    for subtree in tree3.subtrees(filter=lambda t: t.label() == 'VS'):
        g3_chunks.append(subtree)
    g4_chunks = []
    for subtree in tree4.subtrees(filter=lambda t: t.label() == 'Commas'):
        g4_chunks.append(subtree)

    return g1_chunks,g2_chunks,g3_chunks,g4_chunks


# In[21]:

#Function to combine all the phrases and combine them into a single list
def training_set(chunks):
    '''creates a dataframe that easily parsed with the chunks data '''
    df = pd.DataFrame(chunks)    
    df.fillna('X', inplace = True)
    
    train = []
    for row in df.values:
        phrase = ''
        for tup in row:
            # needs a space at the end for seperation
            phrase += tup[0] + ' '
        phrase = ''.join(phrase)
        # could use padding tages but encoder method will provide during 
        # tokenizing/embeddings; X can replace paddding for now
        train.append( phrase.replace('X', '').strip())

    df['phrase'] = train

    # only returns 10% of each dataframe to be used 
    return df.phrase


# In[22]:

#Create Softmax object
y=torch.nn.Softmax()


# In[23]:

#Load the Jobs Data Set
jdskill=pd.read_csv('JDToSkills.csv',encoding='latin-1')


# In[24]:

#get array of skills from String
def getSkillArray(arr):
    arrSplit=arr.split(",")
    newArr=[]
    for i in arrSplit:
        i=clean_text(i)
        newArr.append(i)    
    return newArr 

#Get jobs based on disability of the user
def returnDF(disability,df):
    print(disability)
    print(df)
    getIndex=[]
    for index, row in df.iterrows():
        if len(row['Disability'])>0:
            if disability in row['Disability']:
                getIndex.append(index)
    newTestDf=jdskill.iloc[getIndex]
    return newTestDf

#Recommendation Engine
class remommendationModel:
    def __init__(self, skills, df):
        self.dataframe = df
        self.skill_set = skills
        
    def recommendation_vectors(self):
        final_list = []
        #test1=self.skill_set
        #skillArray1=getSkillArray(self.skill_set)
        counter1 = Counter(self.skill_set)
        for index, row in self.dataframe.iterrows():
            if(len(row['Skills']) > 0):
                skillArray2=getSkillArray(row['Skills'])
                counter2 = Counter(skillArray2)
                all_items = set(counter1.keys()).union( set(counter2.keys()) )
                vector1 = [counter1[k] for k in all_items]
                vector2 = [counter2[k] for k in all_items]
                similarity = 1 - spatial.distance.cosine(vector1, vector2)
                if(similarity > 0):
                    #similarity = cluster.util.cosine_distance(vector1,vector2)
                    new_dict = [row['jobTitle'], row['company'], row['Skills'],row['Disability'],similarity]
                    final_list.append(new_dict)
        return final_list

#Get Skills based on user bio
def getSkills(df_bio,embeddingDim,vocab,pretrained_weights,en_model):
    text1 = df_bio.lower()
    text1 = clean_text(text1)
    pos_tagged_arrs = pos_series(text1)
    g1_chunks,g2_chunks,g3_chunks,g4_chunks=grammar(pos_tagged_arrs)
    training = pd.concat([training_set(g1_chunks),
                      training_set(g2_chunks), 
                      training_set(g3_chunks),training_set(g4_chunks)], 
                        ignore_index = True )
    model=getModel(embeddingDim,vocab,pretrained_weights)
    skills=[]
    for j in training:
        output=y(torch.tensor(predictSentiment(j,model,en_model,vocab)))
        if int(output.argmax().numpy())==1:
            j=clean_text(j)
            j=j.replace(", ","")
            j=j.replace(" ,","")
            skills.append(j)
    return skills



# In[4]:

#API Code
#Loading of Libraries
from fastapi import FastAPI,Form
from pydantic import BaseModel

#Creating API object
app = FastAPI()


#Create UserInput Object. This is used as an Input Request JSON
class UserInput(BaseModel):
    Bio: str=Form()
    disability: str=Form()
        

import json

# In[ ]:
@app.get('/')
async def index():
    return {"Message": "This is Index"}

#POST API End Point Exposed.
@app.post('/predictJobs/')
async def predict(UserInput: UserInput):
    #prediction = MODEL.predict([UserInput.user_input])
    en_model,vocab,pretrained_weights=getWord2Vec(embeddingDim)
    skills=getSkills(UserInput.Bio,embeddingDim,vocab,pretrained_weights,en_model)
    finalDF=returnDF(UserInput.disability,jdskill)
    model_class = remommendationModel(skills, finalDF)
    recommendation_df = model_class.recommendation_vectors()
    recommendation_df = sorted(recommendation_df, key=lambda x: x[4], reverse=True)
    filtered_profiles = pd.DataFrame(recommendation_df, columns=['jobTitle', 'company', 'Skills','Disability','similarity'])
    relevent_profiles = filtered_profiles[['jobTitle', 'company', 'Skills','Disability']]
    #st.dataframe(relevent_profiles.head(5))
    jsonObj=relevent_profiles.head(5).to_json(orient ='table',indent=4)
    return jsonObj

