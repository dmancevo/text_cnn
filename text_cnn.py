import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Flatten,Dense,Dropout

class Text_CNN:

	def __init__(
		self,
		dim        =8,
		char_window=7,
		emb_epochs =10,
		window     =7,
		chars      =100,
		depth      =6,
		pool_size  =2,
		hidden     =32,
		drp_rate   =.5,
		epochs     =1,
		batch_size =32,
		):

		'''
		dim       : character embeddings dimensionality.
		window    : size of context window covering a specific word.
		chars     : number of characters to read from the message.
		depth     : number of convolutional layers.
		pool_size : max pooling factor.
		hidden    : number of hidden units between fully connected layers.
		drp_rate  : dropout rate between fully connected layers.
		'''

		self.dim         =dim
		self.char_window =char_window
		self.emb_epochs  =emb_epochs
		self.window      =window
		self.chars       =chars
		self.depth       =depth
		self.pool_size   =pool_size
		self.hidden      =hidden
		self.drp_rate    =drp_rate
		self.epochs      =epochs
		self.batch_size  =batch_size

	def _char2vec(self,docs):
		S=np.zeros(shape=(len(docs),self.chars,self.dim))
		for i in range(len(docs)):
			for j in range(min(self.chars, len(docs[i]))):
				try:
					S[i,j,:] = self.char2vec.wv[docs[i][j]]
				except KeyError:
					continue
		return S

	def fit(self,X,y):
		'''
		X: list of documents where each document is represented by a string.
		y: document labels for classification.
		'''

		self.char2vec = Word2Vec(size=self.dim, window=self.char_window, min_count=1, workers=4)

		self.char2vec.build_vocab(X)
		self.char2vec.train(X,total_examples=self.char2vec.corpus_count,epochs=self.emb_epochs)

		f = 2
		self.cnn = Sequential([
			Conv1D(
				filters=f,
				kernel_size=self.window, padding='same', input_shape=(self.chars,self.dim)),
			MaxPooling1D(),
		])

		for c in range(2,self.depth+1):
			f *= 2
			self.cnn.add(Conv1D(filters=f, kernel_size=self.window, padding='same'))
			self.cnn.add(MaxPooling1D(pool_size=self.pool_size))

		self.cnn.add(Flatten())
		self.cnn.add(Dense(self.hidden,activation='relu'))
		self.cnn.add(Dropout(self.drp_rate))
		self.cnn.add(Dense(2,activation='softmax'))
		self.cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
		self.cnn.fit(self._char2vec(X), y, epochs=self.epochs, batch_size=self.batch_size)

	def predict(self, X):
		'''
		Return per-class probability.
		'''
		return np.argmax(self.cnn.predict(self._char2vec(X)),axis=1)

	def predict_proba(self, X):
		return self.cnn.predict(self._char2vec(X))

	def save(self, foldername='saved'):
		from os import makedirs
		makedirs(foldername)
		self.cnn.save('{0}/txt_cnn.h5'.format(foldername))
		self.char2vec.save('{0}/char2vec.model'.format(foldername))

	def load(self, foldername='saved'):
		from keras.models import load_model
		self.char2vec = Word2Vec.load('{0}/char2vec.model'.format(foldername))
		self.cnn      = load_model('{0}/txt_cnn.h5'.format(foldername))
