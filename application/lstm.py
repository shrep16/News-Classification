from keras.models import load_model
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class LSTMModel(object):
	def __init__(self):
		self.model = load_model('LSTMmodel.h5')
		self.df = pd.read_csv('preprocess.csv')
		self.df.category = self.df.category.astype(str)
		self.df.headline = self.df.headline.astype(str)
		self.find_labels()
		self.getTokenizer()

	def find_labels(self):
		labelSet = set()
		for category in self.df.category:
			labelSet.add(category)
		self.labelList = list(labelSet)

	def getTokenizer(self):
		MAX_NB_WORDS = 50000
		self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
		self.tokenizer.fit_on_texts(self.df['headline'].values)

	def predictCategory(self, user_headline):
		MAX_SEQUENCE_LENGTH = 250
		seq = self.tokenizer.texts_to_sequences(user_headline)
		padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
		pred = self.model.predict(padded)
		result = self.labelList[np.argmax(pred)]
		return self.labelList[np.argmax(pred)]
		
