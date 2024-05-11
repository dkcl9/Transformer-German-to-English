# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:37:58 2024

@author: aidan kim
"""
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import io
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from MyTransformer import MyTransformer
from util.bleu import get_bleu
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

from collections import Counter
from torchtext.vocab import vocab

#add bos and eos on the sentence and tokenize every word
def tokenize_sentence(sentence, vocab, tokenizer):
    token_arr = [vocab['<bos>']]
    for token in tokenizer(sentence.rstrip("\n")):
        if token in vocab.keys():
            token_arr.append(vocab[token])
        else:
            token_arr.append(vocab['<unk>']) #If given word is not in vocab set it to unknown
    token_arr.append(vocab['<eos>'])
    return token_arr

#tokenize all data sentence using given vocab
def tokenize_all_sentences(filepath, vocab, tokenizer):
    tokenized_sentences = []
    with io.open(filepath, encoding="utf8") as filehandle:
        for sentence in filehandle:
            if sentence != "\n":
                tokenized_sentences.append(torch.tensor(tokenize_sentence(sentence, vocab, tokenizer)))
    return tokenized_sentences

#create batch and pad sequences
def create_batch(each_data_batch,PAD_IDX):
     de_batch, en_batch = [], []

     for (de_item, en_item) in each_data_batch:
         
         de_batch.append(de_item)
         
         en_batch.append(en_item)
     
 
     de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
     en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
     return de_batch, en_batch


de_vocab =torch.load('German_vocab.pth', de_tokenizer).stoi
en_vocab = torch.load('English_vocab.pth', en_tokenizer).stoi



class Dataset(Dataset):
    def __init__(self, data1,data2):
        self.data1 = data1
        self.data2 = data2
    
    def __len__(self):
        return len(self.data1)
    
    def __getitem__(self, idx):
        de_item = self.data1[idx]
        en_item = self.data2[idx]
        return (de_item,en_item)

def create_memory_key_padding_mask(src, PAD_IDX):
    # Create a mask where True values correspond to padding tokens
    memory_key_padding_mask = (src == PAD_IDX)
    return memory_key_padding_mask.transpose(0, 1) 

#Class to perform early stopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='saved_model.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        #Saves model when validation loss decrease.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



#tokenize datasets
de_tokenized_dataset = tokenize_all_sentences('data/train.de', de_vocab, de_tokenizer)
en_tokenized_dataset = tokenize_all_sentences('data/train.en', en_vocab, en_tokenizer)
de_tokenized_val = tokenize_all_sentences('data/val.de', de_vocab, de_tokenizer)
en_tokenized_val = tokenize_all_sentences('data/val.en', en_vocab, en_tokenizer)
de_tokenized_test = tokenize_all_sentences('data/test.de', de_vocab, de_tokenizer)
en_tokenized_test = tokenize_all_sentences('data/test.en', en_vocab, en_tokenizer)

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']

train_data = Dataset(de_tokenized_dataset,en_tokenized_dataset)
train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: create_batch(x, PAD_IDX))

val_data = Dataset(de_tokenized_val,en_tokenized_val)
val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: create_batch(x, PAD_IDX))

test_data = Dataset(de_tokenized_test,en_tokenized_test)

NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512  # Embedding dimension
NHEAD = 8  # Number of Attention heads
FFN_HID_DIM = 512  # Feedforward dimension
SRC_VOCAB_SIZE = len(de_vocab)
TGT_VOCAB_SIZE = len(en_vocab)

model = MyTransformer(NUM_ENCODER_LAYERS,
 NUM_DECODER_LAYERS,
 EMB_SIZE, NHEAD,
 SRC_VOCAB_SIZE,
 TGT_VOCAB_SIZE,
 FFN_HID_DIM)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = torch.optim.Adam(
model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Function to train model on epoch and return training loss
def train_epoch(model,train_iter,optimizer):
    total_loss = 0
    model.train()
    print('-----training----')
    i = 0
    for src, tgt in train_iter:
        i += 1
        
        src = src.to(device)
        tgt = tgt.to(device)
        src_padding_mask = create_memory_key_padding_mask(src, PAD_IDX)
        logits = model.forward(src,tgt[0:-1,:],src_padding_mask,torch.tensor(PAD_IDX))
        optimizer.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(str(i) + '/227  processed..'+'  loss: '+str(loss.item()),end='\r', flush=True)
    train_loss = total_loss/len(list(train_iter))
    return train_loss

#Function to evaluate the model on epoch and return validation loss
def val_epoch(model,val_iter):
    model.eval()
    total_loss = 0
    print('-----evaluating----')
    i = 0
    for src, tgt in val_iter:
        i = i+1
        
        src = src.to(device)
        tgt = tgt.to(device)
        src_padding_mask = create_memory_key_padding_mask(src, PAD_IDX)
        logits = model.forward(src,tgt[0:-1,:],src_padding_mask,torch.tensor(PAD_IDX))
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        total_loss += loss.item()
        print(str(i) + '/8  processed..'+'  loss: '+str(loss.item()),end='\r', flush=True)
    val_loss = total_loss/len(list(val_iter))
    return val_loss

#Function to Train, validate and test model and printout loss and bleu score over epochs
def train(model,train_iter,val_iter,optimizer,test_data):
    epoch = 0
    train_losses = []
    val_losses = []
    bleu_scores = []
    early_stopper = EarlyStopping(patience=5, verbose=True, path='saved_model.pt')
    max_epoch = 18
    for e in range(max_epoch):
        epoch +=1
        print("epoch:",epoch)
        train_loss = train_epoch(model,train_iter,optimizer)
        train_losses.append(train_loss)
        val_loss = val_epoch(model,val_iter)
        val_losses.append(val_loss)
        print("train_loss: {}     val_loss: {}".format(train_loss,val_loss))
        early_stopper(val_loss, model)#check whether validation loss is decreased
        bleu = test(test_data)
        bleu_scores.append(bleu)
        if early_stopper.early_stop:#If it pass 5epoch without improvement eraly stop the training
            print("Early stopping")
            break
    print_loss(train_losses,val_losses)
    print_bleu(bleu_scores,epoch)

#Functions to plot train loss and validation loss over epochs
def print_loss(train_losses,val_losses):
    epochs = np.arange(1, len(train_losses)+ 1)
    plt.plot(epochs, train_losses, label = "train_loss")
    plt.plot(epochs, val_losses,label = "val_loss")
    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Loss over epoch')

    plt.show()
    plt.savefig("loss.png")

#Function to plot bleu score over epochs
def print_bleu(bleu_scores,epoch):
    epochs = np.arange(1, len(bleu_scores)+1)
    plt.plot(epochs, bleu_scores)
    plt.xlabel('epoch')

    plt.ylabel('bleu')

    plt.title('bleu_score')

    plt.show()
    plt.savefig("score.png")

#Function to test model using greedy method to predict each token at a time.
#This function returns average bleu score of given testing data
def test(test_data):
    #Get model parameters
    model.load_state_dict(torch.load('saved_model.pt'))
    model.eval()
    total_score = 0
    w = 0
    for i in range(len(test_data)):
        src,tgt = test_data[i]
        length = src.size(dim=0)
        src = src.view(-1, 1)
        src_mask = torch.zeros((length, length)).type(torch.bool)
        src = src.to(device)
        src_mask = src_mask.to(device)
        input_encoder = model.encode(src,src_mask)
        input_encoder = input_encoder.to(device)
        itr = True
        pred = torch.tensor([[0]])
        pred[0,0] = en_vocab['<bos>']
        pred = pred.to(device)
        pred_len = 1
        while pred_len <= length + 5:
            tgt_mask = model.generate_square_subsequent_mask(pred.size(dim=0)).type(torch.bool)
            tgt_mask = tgt_mask.to(device)
            hat = model.decode(pred,input_encoder,tgt_mask)
            decode_out = hat.transpose(0, 1)
            prob = model.generator(decode_out[:, -1])
            value, Yt = torch.max(prob, dim=1)
            Yt = Yt.item()
            next_token = torch.tensor([[0]])
            next_token[0,0] = Yt
            pred = torch.cat([pred,next_token], dim=0)
            pred_len += 1
            if Yt == en_vocab['<eos>']:
                break
        
        #detokenize the predicted sentence
        pred_sent = ""
        pred_list = list(pred.cpu().numpy())
        pred_detoken = []
        for i in range(0,len(pred_list)):
            de_tok =[k for k, v in en_vocab.items() if v == pred_list[i]][0]
            if de_tok  == '<bos>' or de_tok  =='<eos>':
                continue
            else:
                pred_detoken.append(de_tok)
        
        pred_sent = " ".join(pred_detoken)
        w+=1
        #print(pred_sent)

        #detokenize the target sentence
        target_sent = ""
        target_detoken = []
        for i in range(0,tgt.size(dim=0)):
            de_tok = [k for k, v in en_vocab.items() if v == tgt[i].numpy()][0]
            if de_tok  == '<bos>' or de_tok  == '<eos>':
                continue
            else:
                target_detoken.append(de_tok)

        target_sent = " ".join(target_detoken)
        #print(target_sent)

        #Check Bleu score
        score =  get_bleu(hypotheses=pred_sent.split(), reference=target_sent.split())
        print(str(w) + '/500  processed..'+'  score: '+str(score),end='\r', flush=True)
        total_score += score
        
    print("result",total_score/500)   
    return total_score/500



if __name__ == "__main__":
    train(model,train_iter,val_iter,optimizer,test_data)
    test(test_data)




