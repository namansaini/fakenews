
import numpy as np
import pandas as pd
import tensorflow as tf

data=pd.read_csv("fake.csv", usecols=[2,4,5,8,19])

data=data.dropna()

test_data=data.sample(frac=0.3,random_state=20)
train_data=data.drop(test_data.index)

def encode_label(column):
    columns = column.unique()
    dic = {}
    for i, col in enumerate(columns):
        dic[col] = i
    return column.apply(lambda x: dic[x])


train_data.author=encode_label(train_data.author)

train_data.site_url=encode_label(train_data.site_url)

labels=encode_label(train_data.type)



val_data=train_data.sample(frac=0.2,random_state=20)
train_data=train_data.drop(val_data.index)

unlabel_data = train_data.sample(frac=0.5,random_state=20)

label_data = train_data.drop(unlabel_data.index)

unlabel_data = unlabel_data.drop(columns=[19])

def random_batch(batch_size):
    # Number of images in the training-set.
    num_rows = label_data.count

    # Create a random index.
    idx = np.random.choice(num_rows, size=batch_size,replace=False)

    # Use the random index to select random images and labels.
    x_batch = label_data_train[idx, :, :, :]
    y_batch = label_train[idx, :]

    return x_batch, y_batch
