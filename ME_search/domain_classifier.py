#!/usr/bin/env python
# coding: utf-8

# In[1]:


import easydict
import sys
import re
import os
import random
import tarfile
import urllib
import datetime
import torch
import numpy as np
import torchtext
import torchtext.data as data
from torchtext.data import Dataset
import torchtext.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import json
from tqdm import tqdm


# In[2]:


args = easydict.EasyDict({
    "lr":0.001,
    "epochs":20,
    "batch_size":32,
    "log_interval":1,
    "test_interval":1,
    "save_interval":500,
    "save_dir":"domain_classifier",
    "save_best":True,
    "shuffle":False,
    "dropout":0.5,
    "max_norm":3.0,
    "embed_dim":128,
    "kernel_num":100,
    "static":False,
    "no_cuda":False,
    "snapshot":None,
    "predict":None,
    "test":False
})

if torch.cuda.is_available():
    args.device = torch.device('cuda:1')
else:
    args.device = torch.device('cpu')


# In[3]:


class WiseKBData(data.Dataset):
    filename_list = [
                     'food',
                     'weather',
                     'shopping'
                    ]
    dirname = 'data/20191108/'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):
            string = re.sub(r"[^ㄱ-ㅣ가-힣A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            examples = []
            path = self.dirname if path is None else path
            for fileclass in self.filename_list:
                print(fileclass, "ing ..." )

                with open(os.path.join(path, fileclass+'/QM.txt'), encoding = 'utf-8') as fin:
                    txt_data = fin.read().strip().split('\n')

                    for line in txt_data:
                        line = line.split(" ")
                        for j in range(len(line)):
                            temp = line[j].split("/")
                            line[j] = temp[0]
                        line = " ".join(line)

                        examples += [data.Example.fromlist([line, fileclass], fields)]

        super(WiseKBData, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
        """
        examples = cls(text_field, label_field, **kwargs).examples

        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


# In[4]:


# load data
print("Loading data...\n")

text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_data, dev_data = WiseKBData.splits(text_field, label_field)
train_iter, dev_iter = data.Iterator.splits(
                            (train_data, dev_data),
                            batch_sizes=(args.batch_size, args.batch_size),
                            device=args.device, repeat=False)

text_field.build_vocab(train_data, dev_data)
label_field.build_vocab(train_data, dev_data)


print("\nFinished...")


# In[5]:


args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# In[6]:


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.Co = args.kernel_num
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        self.sentence_max_size = 40

        self.conv3 = nn.Conv2d(1, 1, (3, D))
        self.conv4 = nn.Conv2d(1, 1, (4, D))
        self.conv5 = nn.Conv2d(1, 1, (5, D))
        self.Avg3_pool = nn.AvgPool2d((self.sentence_max_size-3+1, 1))
        self.Avg4_pool = nn.AvgPool2d((self.sentence_max_size-4+1, 1))
        self.Avg5_pool = nn.AvgPool2d((self.sentence_max_size-5+1, 1))
        self.linear1 = nn.Linear(3, self.C)

    def forward(self, x):
        x = F.pad(x, pad=(0, self.sentence_max_size-x.shape[1]), mode='constant', value=1)
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)

        batch = x.shape[0]
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        a1 = self.Avg3_pool(x1)
        a2 = self.Avg4_pool(x2)
        a3 = self.Avg5_pool(x3)

        x = torch.cat((a1, a2, a3), -1)
        x = x.view(batch, 1, -1)

        x = self.linear1(x)
        x = x.view(-1, self.C)

        return x, x1, x2, x3


# In[37]:


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature = feature.data.t()
            target = target.data.sub(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, _, _, _ = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data,
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))


        dev_acc = eval(dev_iter, model, args)

        if dev_acc > best_acc:
            best_acc = dev_acc
            last_step = steps
            if args.save_best:
                save(model, args.save_dir, 'best', steps)


# In[38]:


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()
        target = target.data.sub(1)  # batch first, index align

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit, _, _, _ = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum')

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\rEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


# In[39]:


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


# In[40]:


cnn = TextCNN(args)
cnn = cnn.to(args.device)
train(train_iter, dev_iter, cnn, args)
