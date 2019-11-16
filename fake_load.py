#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:53:34 2019

@author: mukultanwar
"""


import numpy as np
import pandas as pd
import tensorflow as tf

data=pd.read_csv("fake.csv", usecols=[2,4,5,8,19])

data=data.dropna()

test_data=data.sample(frac=0.3,random_state=200)
train_data=data.drop(test_data.index)


authors=train_data.author.unique()
dic={}
for i,auth in enumerate(authors):
    dic[auth]=i
train_data.author=train_data.author.apply(lambda x:dic[x])
sites=train_data.site_url.unique()
dic={}
for i,site in enumerate(sites):
    dic[site]=i
train_data.site_url=train_data.site_url.apply(lambda x:dic[x])

types=train_data.type.unique()
dic={}
for i,type in enumerate(types):
    dic[type]=i
labels=train_data.type.apply(lambda x:dic[x])



val_data=train_data.sample(frac=0.2,random_state=200)
train_data=train_data.drop(val_data.index)

unlabel_data = train_data.sample(frac=0.5,random_state=200)

label_data = train_data.drop(unlabel_data.index)

unlabel_data = unlabel_data.drop(columns=[19])

