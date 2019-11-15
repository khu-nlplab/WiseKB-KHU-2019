#!/usr/bin/env python
# coding: utf-8

# In[74]:


from nltk.classify import maxent
import nltk
import sys
import io
import os
import numpy
from random import shuffle
import urllib3
import json
import ast
import alignment
import torch.nn as nn
import torch.nn.functional as F
import torch
import easydict

import torchtext
import torchtext.data as data
import torchtext.datasets as datasets

import datetime
import pickle
import dill
import re
import timeit
import operator

args = easydict.EasyDict({
    "embed_num":3257,
    "embed_dim":128,
    "kernel_num":100,
    "class_num":3
})

FOOD_QM = "data/food/MEM data/QM.txt"
SHOPPING_QM = "data/shopping/MEM data/QM.txt"
WEATHER_QM = "data/weather/MEM data/QM.txt"

FOOD_POS = "data/food/MEM data/positive_org.txt"
FOOD_NEG = "data/food/MEM data/negative_org.txt"
SHOPPING_POS = "data/shopping/MEM data/positive_org.txt"
SHOPPING_NEG = "data/shopping/MEM data/negative_org.txt"
WEATHER_POS = "data/weather/MEM data/positive_org.txt"
WEATHER_NEG = "data/weather/MEM data/negative_org.txt"

if torch.cuda.is_available():
    args.device = torch.device('cuda:1')
else:
    args.device = torch.device('cpu')


def sentence_parse(tgt_sentence):
    openApiURL :str = "http://aiopen.etri.re.kr:8000/WiseNLU"
    analysisCode :str = "dparse" # morp, dparse
    requestJson = {
        "access_key": "your key", # 
        "argument": {
            "text": '',
            "analysis_code": analysisCode
        }
    }
    result: str = ''

    ### Make http pool & request
    http = urllib3.PoolManager()

    requestJson['argument']['text'] = tgt_sentence
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )

    jsonStr = json.loads(str(response.data, "utf-8"))

    result_list = []
    for morp in jsonStr['return_object']['sentence'][0]['morp']:
        result_list.append(morp["lemma"] + "/" + morp["type"])

    result = " ".join(result_list)

    return result


# In[36]:


def data_preprocessing1(_list):
    temp_list = []
    temp_list.extend(_list)
    for i in range(len(temp_list)):
        temp = temp_list[i]
        if "EF" not in temp and "EC" not in temp and "ET" not in temp and "EP" not in temp and "XS" not in temp and "JX" not in temp and "JC" not in temp:
            temp = temp.split('/')
            temp_list[i] = temp[1]
    return temp_list


# In[37]:


def data_preprocessing2(origin_list, processed_list):
    j = 0
    for i in range(len(processed_list)):
        if processed_list[i] == '-':
            processed_list[i] = '-/-'
        else:
            processed_list[i] = origin_list[j]
            j = j+1

    return processed_list


# In[38]:


def add_processing(a, b):
    result_list = []
    for i in range(len(a)):
        temp = (a[i], b[i])
        result_list.append(temp)

    return result_list


# In[51]:


def load_data(org_route, target_route):
    original_list = open(org_route,  encoding = 'utf-8').read().strip().split('\n')
    data_list = open(target_route,  encoding = 'utf-8').read().strip().split('\n')
    result_list = []

    for i in range(len(data_list)):
        pair = data_list[i].split(",")

        src1 = original_list[int(pair[0])].split(" ")
        src2 = original_list[int(pair[1])+250].split(" ")
        p_src1 = data_preprocessing1(src1)
        p_src2 = data_preprocessing1(src2)


        dst1 = alignment.needle(p_src1, p_src2)[2]
        dst2 = alignment.needle(p_src1, p_src2)[3]
        dst1 = data_preprocessing2(src1, dst1)
        dst2 = data_preprocessing2(src2, dst2)


        temp = add_processing(dst1, dst2)
        result_list.append(temp)

    return result_list


# In[52]:


def make_feature(_list, feature_class):
    result_list = []
    for i in range(len(_list)):
        features = {}
        features['lexical'] = 0
        features['morpheme'] = 0
        features['eomal_eomi'] = 0
        features['seoneomal_eomi'] = 0
        features['jeopmisa'] = 0
        features['kyeokjosa'] = 0
        features['bojosa'] = 0

        for j in range(len(_list[i])):
            first, second = _list[i][j][0], _list[i][j][1]
            first_lexical, first_morph = first.split('/')
            second_lexical, second_morph = second.split('/')

            if first_lexical == second_lexical:
                features['lexical'] += 1
            if first_morph == second_morph:
                features['morpheme'] += 1

                if "JK" in first_morph:
                    features['kyeokjosa'] += 1
                elif first_lexical == second_lexical:
                    if "EF" in first_morph or "EC" in first_morph or "ET" in first_morph:
                        features['eomal_eomi'] += 1
                    elif "EP" in first_morph:
                        features['seoneomal_eomi'] += 1
                    elif "XS" in first_morph:
                        features['jeopmisa'] += 1
                    elif "JX" in first_morph:
                        features['bojosa'] += 1

        for key in features:
            features[key] = 0.1 if features[key] == 0 else features[key]

        if len(feature_class) > 0:
            result_list.append((features, feature_class))
        else:
            result_list.append((features))

    return result_list


# In[69]:
def train_all_maxent():
    food_classifier = train_maxent(FOOD_QM, FOOD_POS, FOOD_NEG)
    shopping_classifier = train_maxent(SHOPPING_QM, SHOPPING_POS, SHOPPING_NEG)
    weather_classifier = train_maxent(WEATHER_QM, WEATHER_POS, WEATHER_NEG)

    return food_classifier, shopping_classifier, weather_classifier

def train_maxent(org_data_route, pos_data_route, neg_data_route):
    pos_data = load_data(org_data_route, pos_data_route)
    neg_data = load_data(org_data_route, neg_data_route)

    train_data = []
    test_data = []
    train_data.extend(make_feature(pos_data, "P"))
    train_data.extend(make_feature(neg_data, "N"))

    try:
        encoding = nltk.classify.maxent.TypedMaxentFeatureEncoding.train(train_data)
        weights = numpy.zeros(encoding.length(), 'd')
        classifier = nltk.classify.maxent.MaxentClassifier.train(train_data, encoding = encoding, algorithm='iis', trace = 0)

    except Exception as e:
        print('Error: %r' % e)
        return

    return classifier


# In[54]:


class NewData(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, input_text, path=None, examples=None, **kwargs):
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
        fields = [('text', text_field)]

        examples = []
        examples += [data.Example.fromlist([input_text], fields)]

        super(NewData, self).__init__(examples, fields, **kwargs)


# In[55]:


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.Co = args.kernel_num
        V = args.embed_num
        D = args.embed_dim
        self.C = args.class_num
        self.embed = nn.Embedding(V, D)
        self.sentence_max_size = 50

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



def domain_classify(dc_model, text_field, label_field, input_text):
    test_data = NewData(text_field, input_text)
    test_iter = data.Iterator(test_data, batch_size=1)

    for batch in test_iter:
        feature = batch.text
        feature = feature.data.t()
        logit, _, _, _ = dc_model(feature)

    range_size = len(torch.max(logit, 1)[1].view(torch.Size([1])).data)

    for i in range(range_size):
        label = label_field.vocab.itos[torch.max(logit, 1)[1].view(torch.Size([1])).data[i]+1]

    return label


# In[58]:


def detach_morp_tag(input_text):
    input_text = input_text.split(" ")

    for j in range(len(input_text)):
        temp = input_text[j].split("/")
        input_text[j] = temp[0]

    input_text = " ".join(input_text)

    return input_text


# In[59]:


def extract_simliar_sentence(label, input_text, extract_num, classifier):
    temp_query = input_text.split(" ")
    query = data_preprocessing1(temp_query)

    if label == "food":
        org_data_list = open(FOOD_QM,  encoding = 'utf-8').read().strip().split('\n')
    elif label == "shopping":
        org_data_list = open(SHOPPING_QM,  encoding = 'utf-8').read().strip().split('\n')
    elif label == "weather":
        org_data_list = open(WEATHER_QM,  encoding = 'utf-8').read().strip().split('\n')

    data_list = []
    for i in range(len(org_data_list)):
        temp_data = org_data_list[i].split(" ")
        p_data = data_preprocessing1(temp_data)
        data_list.append(p_data)

    result_list = []
    for j in range(len(data_list)):
        dst1 = alignment.needle(query, data_list[j])[2]
        dst2 = alignment.needle(query, data_list[j])[3]
        dst1 = data_preprocessing2(input_text.split(" "), dst1)
        dst2 = data_preprocessing2(org_data_list[j].split(" "), dst2)

        temp = add_processing(dst1, dst2)
        result_list.append(temp)

    test_list = make_feature(result_list, "")
    score_dict = dict()

    for i in range(len(test_list)):
        score = classifier.prob_classify(test_list[i]).prob('P')
        score_dict[i] = score

    extract_list = []
    count = 0
    for i in sorted(score_dict, key=lambda m: score_dict[m], reverse=True):
        extract_list.append(input_text + ":" + org_data_list[i])
        count += 1
        if count >= extract_num:
            break

    return extract_list
