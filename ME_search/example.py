#!/usr/bin/env python
# coding: utf-8

# In[1]:


import module
import torch
import dill


# In[2]:


cnn = module.TextCNN(module.args)


# In[3]:


cnn.load_state_dict(torch.load("dc_model.pt"))


# In[4]:


cnn.to(module.args.device)


# In[5]:


with open("dc_text3.pt","rb")as f:
    text_field = dill.load(f)
with open("dc_label3.pt","rb")as f:
    label_field = dill.load(f)


# In[6]:


input_text = "허리 가 아프 ㄴ 게 밖 에 날씨 가 않 좋 나 ?"


# In[7]:


input_morp = module.sentence_parse(input_text)


# In[8]:


label = module.domain_classify(cnn, text_field, label_field, module.detach_morp_tag(input_morp))


# In[9]:


food_classifier, shopping_classifier, weather_classifier = module.train_all_maxent()


# In[10]:


if label == "food" :
    classifier = food_classifier
    
elif label == "shopping" :
    classifier = shopping_classifier
    
elif label == "weather" :
    classifier = weather_classifier


# In[11]:


output_morp_list = module.extract_simliar_sentence(label, input_morp, 10, classifier)


# In[12]:


print(output_morp_list)

