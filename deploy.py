
from flask import Flask, request, render_template
import requests
import sys
# sys.path.append('/content/trac2020_submission/src/')
import collections
from typing import Callable
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import notebook
import importlib
import pprint
import nltk
import datetime
import os
from argparse import Namespace
import re
from collections import Counter

import utils.general as general_utils
import utils.trac2020 as trac_utils
import utils.transformer.data as transformer_data_utils
import utils.transformer.general as transformer_general_utils
general_utils.set_seed_everywhere() #set the seed for reproducibility


import logging
logging.basicConfig(level=logging.INFO) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Import RAdam and Lookahead
from radam.radam import RAdam
from lookahead.optimizer import Lookahead

from transformers import XLMRobertaTokenizer, XLMRobertaModel

args = Namespace(
        #use cuda by default
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
        #set batch size and number of epochs
        batch_size = 32,
        num_epochs = 20,
    
        #set the learning rate
        learning_rate = 0.0001,

        #location of the train, dev and test csv
        train_csv = 'final_train_cleaned.csv',
        dev_csv = 'trac2_hin_dev.csv',
        test_csv = 'trac2_hin_test.csv',
    
        #directory to save our models at
        directory = './', 
        model_name = 'xlmrobeta_hin_b.pt',
)

raw_train_df =  pd.read_csv(args.train_csv)
raw_train_df['split'] = 'train'
raw_dev_df =  pd.read_csv(args.dev_csv)
raw_dev_df['split'] = 'dev'

# Concatinate both train and dev dfs together
data_df = pd.concat([raw_dev_df, raw_train_df], ignore_index= True)
task_b_label_dict = {'NGEN':0, 'GEN':1}
data_df_task_b = data_df[['ID','Text','Sub-task B','split']].copy()
data_df_task_b.columns.values[1] = 'text'
data_df_task_b.columns.values[2] = 'label'
data_df_task_b.loc[:,'label'] = data_df_task_b.loc[:,'label'].map(task_b_label_dict) 
#split long sentences into sentences of 200 words
data_df_task_b['text'] = data_df_task_b['text'].map(lambda x: trac_utils.chunk_sent(x,150,50))
exploded_df = data_df_task_b.explode('text').reset_index()

class RobertaPreprocessor():
    """
    Preprocessor for adding special tokens into each sample
    NOTE: Doesn't work perfectly.
    """
    
    
    def __init__(self,transformer_tokenizer,sentence_detector):
        """
        Args:
            transformer_tokenizer: Tokenizer for the transformer model
            sentence_detector: Sentence tokenizer.
        """
        self.transformer_tokenizer = transformer_tokenizer
        self.sentence_detector = sentence_detector
        self.bos_token = transformer_tokenizer.bos_token
        self.sep_token = ' ' + transformer_tokenizer.sep_token + ' '
        
    def add_special_tokens(self, text):
        """
        Adds '</s>' between each sentence and at the end of the sample.
        Adds '<s>' at the start of the sentence.
        
        Args:
            text: Text sample to add special tokens into
        Returns:
            text with special tokens added
        """
        text = ' '.join(text.strip().split()) #clean whitespaces
        sentences = self.sentence_detector.tokenize(text)
        eos_added_text  = self.sep_token.join(sentences) 
        return self.bos_token +' '+ eos_added_text + ' ' + self.transformer_tokenizer.sep_token
    
    
    
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
punkt_sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')

roberta_preproc = RobertaPreprocessor(xlmroberta_tokenizer, punkt_sentence_detector)

#apply the preprocessor on the exploded dataframe
exploded_df['text'] = exploded_df['text'].map(roberta_preproc.add_special_tokens)

class SimpleVectorizer():
    """Vectorizes Class to encode the samples into 
    their token ids and creates their respective attention masks
    """
    
    def __init__(self,tokenizer: Callable, max_seq_len: int):
        """
        Args:
            tokenizer (Callable): transformer tokenizer
            max_seq_len (int): Maximum sequence lenght 
        """
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def vectorize(self,text :str):
        """
        Args:
            text: Text sample to vectorize
        Returns:
            ids: Token ids of the 
            attn: Attention masks for ids 
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False, #already added by preprocessor
            max_length = self._max_seq_len,
            pad_to_max_length = True,
        )
        ids =  np.array(encoded['input_ids'], dtype=np.int64)
        attn = np.array(encoded['attention_mask'], dtype=np.int64)
        
        return ids, attn
    
class TracDataset(Dataset):
    """PyTorch dataset class"""
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer: Callable,
        max_seq_length:int = None,
    ):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (Callable): tokenizer for the transformer
            max_seq_length (int): Maximum sequece length to work with.
        """
        self.data_df = data_df
        self.tokenizer = tokenizer

        if max_seq_length is None:
            self._max_seq_length = self._get_max_len(data_df,tokenizer)
        else:
            self._max_seq_length = max_seq_length

        self.train_df = self.data_df[self.data_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == 'dev']
        self.val_size = len(self.val_df)

        self.test_df = self.data_df[self.data_df.split == 'test']
        self.test_size = len(self.test_df)
        
        self._simple_vectorizer = SimpleVectorizer(tokenizer, self._max_seq_length)
        
        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    
    def _get_max_len(self,data_df: pd.DataFrame, tokenizer: Callable):
        """Get the maximum lenght found in the data
        Args:
            data_df (pandas.DataFrame): The pandas dataframe with the data
            tokenizer (Callable): The tokenizer of the transformer
        Returns:
            max_len (int): Maximum length
        """
        len_func = lambda x: len(self.tokenizer.encode_plus(x)['input_ids'])
        max_len = data_df.text.map(len_func).max() 
        return max_len

    
    def set_split(self, split="train"):
        """selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    
    def __len__(self):
        return self._target_size
    
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        
        indices, attention_masks = self._simple_vectorizer.vectorize(row.text)


        label = row.label
        return {'x_data': indices,
                'x_attn_mask': attention_masks,
                'x_index': index,
                'y_target': label}
    
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
    
    
    
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=False, device="cpu", pinned_memory = False, n_workers = 0): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            pin_memory= pinned_memory,
                            num_workers = n_workers,
                            )
    
    for data_dict in dataloader:
        out_data_dict = {}
        out_data_dict['x_data'] = data_dict['x_data'].to(
            device, non_blocking= (True if pinned_memory else False) 
        )
        out_data_dict['x_attn_mask'] = data_dict['x_attn_mask'].to(
            device, non_blocking= (True if pinned_memory else False) 
        )
        out_data_dict['x_index'] = data_dict['x_index']
        out_data_dict['y_target'] = data_dict['y_target'].to(
            device, non_blocking= (True if pinned_memory else False) 
        )
        yield out_data_dict
        
        
dataset = TracDataset(
    data_df = exploded_df,
    tokenizer = xlmroberta_tokenizer,
    max_seq_length = 403 #what we used
)

class XLMRoBertAttention(nn.Module, ):
    """Implements Attention Head Classifier
    on Pretrained Roberta Transformer representations.
    Attention Head Implementation : https://www.aclweb.org/anthology/P16-2034/
    """
    
    def penalized_tanh(self,x):
        """
        http://aclweb.org/anthology/D18-1472
        """
        alpha = 0.25
        return torch.max(torch.tanh(x), alpha*torch.tanh(x))
    
    
    def __init__(self, model_name, num_labels):
        """
        Args:
            model_name: model name, eg, roberta-base'
            num_labels: number of classes to classify
        """
        super().__init__()
        self.w = nn.Linear(768,1, bias=False)
        self.bert = XLMRobertaModel.from_pretrained(model_name)
        self.prediction_layer = nn.Linear(768, num_labels)
        self.init_weights()
        
        
    def init_weights(self):
        """Initializes the weights of the Attention head classifier"""
        
        for name, param in self.prediction_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.w.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        
    def forward(self, input_ids,attention_mask):
        """
        Args:
            input_ids: sent encoded into indices
            attention_mask: their respective attention masks
        Returns:
            preds: Final layer output of the model
        """
        embeddings = self.bert(input_ids = input_ids,
                  attention_mask = attention_mask)
        H = embeddings[0] #final hidden layer outputs 
        M = self.penalized_tanh(H)
        alpha = torch.softmax(self.w(M), dim=1)
        r = torch.bmm(H.permute(0,2,1),alpha)
        h_star = self.penalized_tanh(r)
        preds = self.prediction_layer(h_star.permute(0,2,1))
        return preds
model = XLMRoBertAttention(
    model_name = 'xlm-roberta-base',
    num_labels = 2#len(set(dataset.data_df.label)),
)
model.to(args.device) #send the model to the 'cpu' or 'gpu'

train_state = general_utils.make_train_state() #dictionary for saving training routine information
train_state.keys()

def sort_preds(indexes, preds):
    """Sorts the predictions in order, to reverse the effects of shuffle
    done by dataloader"""
    indexes = indexes.cpu().numpy().reshape(-1,1)
    preds = preds.cpu().numpy()
    arr_concat = np.hstack((indexes,preds)) #concat the preds and their indexes
    sort_arr = arr_concat[ arr_concat[:,0].argsort()] #sort based on the indexes
    sorted_preds = np.delete(sort_arr,0,axis=1)
    return sorted_preds

def get_optimal_models(train_state, split, reverse=False ):
    """Naive Ensembling"""
    trgts= sort_preds(train_state[f'{split}_indexes'][-1],train_state[f'{split}_targets'][-1].reshape(-1,1))
    total_preds = len(train_state[f'{split}_indexes'])
    init = np.zeros(train_state[f'{split}_preds'][-1].shape)
    max_f1 = 0
    idxes = []
    rng = range(0,total_preds)
    if reverse:
        rng = reversed(rng)
    for i in rng:
        temp = sort_preds(train_state[f'{split}_indexes'][i],train_state[f'{split}_preds'][i])
        temp2 = init+temp
        f1 = f1_score(
            y_pred=temp2.argmax(axis=1),
            y_true= trgts, average ='weighted'
        )
        if f1 > max_f1:
            max_f1 = f1
            init = init+temp
            idxes.append(i)
    print(f'Taking preds from {idxes} | Dev f1:{f1}')
    return (idxes,max_f1)


# all_models= [os.path.join(args.directory,i) for i in os.listdir(args.directory) if args.model_name in i]
# all_models = sorted(all_models, key = lambda x: int(x[8])) #sort by epoch num.
# selected_models = all_models

selected_models=['./_epoc_4_xlmrobeta_hin_b.pt',
 './_epoc_5_xlmrobeta_hin_b.pt',
 './_epoc_6_xlmrobeta_hin_b.pt']
# selected_models=['./_epoc_0_xlmrobeta_hin_b.pt',
#  './_epoc_1_xlmrobeta_hin_b.pt',
#  './_epoc_2_xlmrobeta_hin_b.pt',
#  './_epoc_3_xlmrobeta_hin_b.pt',
#  './_epoc_4_xlmrobeta_hin_b.pt',
#  './_epoc_5_xlmrobeta_hin_b.pt',
#  './_epoc_6_xlmrobeta_hin_b.pt']


import string
import re
from fuzzywuzzy import fuzz





app = Flask(__name__)




@app.route('/', methods=['GET', 'POST'])
def index():
 	return render_template('home.html')
@app.route('/process',methods=['GET', 'POST'])
def nextFn():
    
    user_text= request.form.get('raw')
    splits= user_text.split(' ')
    for s in splits:
        ns="".join(sorted(set(s)))


        my_str = ns

        chars = re.escape(string.punctuation)
        nns= re.sub(r'['+chars+']', '',my_str)  
        if fuzz.ratio(nns,"".join(sorted('randi')))>=70:
            result="Profane"
            return render_template('response.html', result=result)
        
    for s in splits:
        ns="".join(sorted(set(s)))


        my_str = ns

        chars = re.escape(string.punctuation)
        nns= re.sub(r'['+chars+']', '',my_str)  
        if fuzz.ratio(nns,"".join(sorted('sex')))>=70:
            result="Profane"
            return render_template('response.html', result=result) 
        
    abdf = pd.read_csv('doubtnut.profanitybank - doubtnut.profanitybank.csv')
    ab_list= abdf['word'].values.tolist()
    for i in ab_list:
        if fuzz.ratio(user_text,i)>=70:
            result="Profane"
            return render_template('response.html', result=result) 
    # test_set_loc = '02072020-profane_eg_to_test.csv'
    test_df = pd.DataFrame({0:[user_text]})
    test_df.columns=['text']
    test_df['text'] = test_df['text'].map(roberta_preproc.add_special_tokens)
    
    test_df['split'] = 'test'  #dummy label
    test_df['label'] = -1  #dummy label
    
    test_dataset = TracDataset(
    data_df = test_df,
    tokenizer = xlmroberta_tokenizer,
    max_seq_length = 403#dataset._max_seq_length
)
    test_dataset.set_split('test')
    test_state = general_utils.make_train_state() 
    test_dataset.set_split('test')
    # test_state = general_utils.make_train_state() 

    eval_bar = notebook.tqdm(
        desc = 'split=train ',
        total=test_dataset.get_num_batches(args.batch_size),
        position=0,
        leave=True,
    )
    model.eval()
    for m in notebook.tqdm(selected_models, total=len(selected_models)):
        eval_bar.reset(
            total=test_dataset.get_num_batches(args.batch_size),
        )
        model.load_state_dict(torch.load(m)['model'])
        batch_generator = generate_batches(
            dataset= test_dataset, batch_size= args.batch_size, shuffle=False,
            device = args.device, drop_last=False,
            pinned_memory = True, n_workers = 10, 
        )
        with torch.no_grad():
            for batch_index, batch_dict in enumerate(batch_generator):
                y_pred = model(
                    input_ids = batch_dict['x_data'],
                    attention_mask =  batch_dict['x_attn_mask'],
                )
                y_pred = y_pred.view(-1, len(set(dataset.data_df.label)))

                y_pred = y_pred.detach()

                batch_dict['y_target'] = batch_dict['y_target'].cpu()
                test_state['batch_preds'].append(y_pred.cpu())
                test_state['batch_targets'].append(batch_dict['y_target'].cpu())
                test_state['batch_indexes'].append(batch_dict['x_index'].cpu())
                eval_bar.update()

        test_state['val_preds'].append(
            torch.cat(test_state['batch_preds']).cpu()
        )
        test_state['val_targets'].append(
            torch.cat(test_state['batch_targets']).cpu()
        )
        test_state['val_indexes'].append(
            torch.cat(test_state['batch_indexes']).cpu()
        )

        test_state['batch_preds'] = []
        test_state['batch_targets'] = []
        test_state['batch_indexes'] = []

    ensemble = torch.zeros_like(test_state['val_preds'][-1])
    for i in test_state['val_preds']:
        ensemble += i
        
    test_preds = torch.argmax(ensemble, dim=1).tolist()
    
        # task_b_label_dict = {'NGEN':0, 'GEN':1} #ref Reading TRAC2020 data... ipynb
    int_to_label = {0:'NGEN', 1:'GEN'}
    pred_labels = [int_to_label[i] for i in test_preds]
    collections.Counter(pred_labels)
    
    
    pred_df = pd.DataFrame( data= {'id':test_df.index, 'label':pred_labels})
    
    pred_analysis_df = pd.DataFrame( data= {'id':test_df.index, 'text':test_df.text ,'label':pred_labels})
    
    pred_analysis_df
    
    p=pred_analysis_df['label'].values.tolist()[0]
    if p=="GEN":
        result="Profane"
        return render_template('response.html', result=result)
    else:
        result="Not Profane"
        return render_template('response.html', result=result)

if __name__=='__main__':
    app.run(debug=True, host='172.31.28.15', port=8891)
