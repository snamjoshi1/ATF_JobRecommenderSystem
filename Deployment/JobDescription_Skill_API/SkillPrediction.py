#!/usr/bin/env python
# coding: utf-8

#Load Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re


# In[12]:

#Set max length
max_seq_len=4


# In[25]:

#Class for BERT Model
class BERT_Arch(nn.Module):

    def __init__(self):
      
      super(BERT_Arch, self).__init__()

      self.bert = AutoModel.from_pretrained('bert-base-uncased')
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x


# In[63]:

#Function to clean data
def clean_text(raw):
    #print(raw)
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
        
        # using try and except due to Nan in the column
    except:
        clean_text = np.nan
        
    return clean_text


# In[74]:

#Function to get POS Tagging for the textual data
def pos_series(keyword):
    '''categorizes after tokenizing words with POS tags'''
    tokens = nltk.word_tokenize(keyword)
    tagged = nltk.pos_tag(tokens)
    return tagged


# In[75]:

#Function to get Phrases from POS Tagged Data
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
    # print(subtree)
    g1_chunks.append(subtree)
  g2_chunks = []
  for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP2'):
    # print(subtree)
    g2_chunks.append(subtree)
  g3_chunks = []
  for subtree in tree3.subtrees(filter=lambda t: t.label() == 'VS'):
    # print(subtree)
    g3_chunks.append(subtree)
  g4_chunks = []
  for subtree in tree4.subtrees(filter=lambda t: t.label() == 'Commas'):
    # print(subtree)
    g4_chunks.append(subtree)

  return g1_chunks,g2_chunks,g3_chunks,g4_chunks


# In[66]:

#Function to combine all the phrases into a single list
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


# In[77]:

#Create a Prediction Class
class Model:
    def __init__(self):
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        classifier = BERT_Arch()
        classifier.load_state_dict(
            torch.load("saved_weights3.pt", map_location="cpu")
        )
        classifier = classifier.eval()
        self.classifier = classifier
    def predict(self, text):
        cleaned_text=clean_text(text)
        cleaned_text = cleaned_text.lower()
        pos_tagged_arrs = pos_series(cleaned_text)
        g1_chunks,g2_chunks,g3_chunks,g4_chunks=grammar(pos_tagged_arrs)
        training = pd.concat([training_set(g1_chunks),
                      training_set(g2_chunks), 
                      training_set(g3_chunks),training_set(g4_chunks)], 
                        ignore_index = True )
        encoded_text = self.tokenizer.batch_encode_plus(
            training.to_list(),
            max_length=max_seq_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True

        )
        data_seq = torch.tensor(encoded_text["input_ids"])
        data_mask = torch.tensor(encoded_text["attention_mask"])
        
        with torch.no_grad():
            preds = self.classifier(data_seq, data_mask)
            preds = preds.detach().numpy()
        preds = np.argmax(preds, axis = 1)
        jd_skills=[]
        for k in range(len(training)):
            if preds[k]==1:
                s=training[k].replace(", ","")
                s=training[k].replace(" ,","")
                jd_skills.append(s)
        return jd_skills
        
        


# In[78]:


#FAST API
#Load Libraries
from fastapi import FastAPI,Form
from pydantic import BaseModel


#Create FAST API object
app = FastAPI()


# In[10]:

#Create UserInput Object. This is used as an Input Request JSON
class UserInput(BaseModel):
    Description: str=Form()
        


# In[ ]:
@app.get('/')
async def index():
    return {"Message": "This is Index"}

#POST API End Point Exposed.
@app.post('/predictSkills/')
async def predict(UserInput: UserInput):
    mod=Model()
    arr1=mod.predict(UserInput.Description)
    return arr1
    

