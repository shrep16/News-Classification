#testing the model

#load the model
from keras.models import load_model
model = load_model('LSTMmodel.h5')
print("model successfully loaded")

#fetch a list of labels/categories
#load data
import pandas as pd

df = pd.read_csv('preprocess.csv')
df.category = df.category.astype(str) #convert to string
df.headline = df.headline.astype(str)

labelSet = set()
for category in df.category:
	labelSet.add(category)

labelList = list(labelSet)

#get a new complaint
MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 50000

new_complaint = ['mass shootings texas last week tv'] #add the complaint 

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
tokenizer.fit_on_texts(df['headline'].values)
from keras.preprocessing.sequence import pad_sequences
seq = tokenizer.texts_to_sequences(new_complaint)

padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)

import numpy as np
print(pred, labelList[np.argmax(pred)])
