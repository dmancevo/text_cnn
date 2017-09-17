from os import makedirs
import numpy as np
from gensim.models import Word2Vec
from keras.engine import Input
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Conv1D,MaxPooling1D,Flatten,Dense,Dropout
from keras.models import load_model
import pickle as pkl

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

	def text_to_matrix(self,X):
		mtrx = self.T.texts_to_matrix(X)[:,:self.chars]
		pad  = np.zeros((mtrx.shape[0],self.chars))
		pad[:mtrx.shape[0],:mtrx.shape[1]] = mtrx
		return pad

	def fit(self,X,y):
		'''
		X: list of documents where each document is represented by a string.
		y: document labels for classification.
		'''

		char2vec = Word2Vec(size=self.dim, window=self.char_window, min_count=1, workers=4)

		char2vec.build_vocab(X)
		char2vec.train(X,total_examples=char2vec.corpus_count,epochs=self.emb_epochs)

		self.T = Tokenizer(filters='',lower=False,split='',char_level=True)
		self.T.fit_on_texts(char2vec.wv.index2word)

		inpt = Input(shape=(self.chars,), dtype='int32')
		x    = char2vec.wv.get_embedding_layer()(inpt)

		f = 2
		for c in range(self.depth):
			
			x = Conv1D(filters=f, kernel_size=self.window, padding='same')(x)
			x = MaxPooling1D(pool_size=self.pool_size)(x)
			f *= 2

		x    = Flatten()(x)
		x    = Dense(self.hidden,activation='relu')(x)
		x    = Dropout(self.drp_rate)(x)
		prob = Dense(len(np.unique(y)),activation='softmax')(x)

		self.cnn = Model(inputs=inpt, outputs=prob)
		self.cnn.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
		self.cnn.fit(self.text_to_matrix(X), y, epochs=self.epochs, batch_size=self.batch_size)

	def predict(self, X):
		'''
		Return most likely class per sample.
		'''
		return np.argmax(self.cnn.predict(self.text_to_matrix(X)),axis=1)

	def predict_proba(self, X):
		'''
		Return per-class probability.
		'''
		return self.cnn.predict(self.text_to_matrix(X))

	def save(self, foldername='saved'):
		'''
		Save model into folder='foldername'
		'''
		makedirs(foldername)
		self.cnn.save('{0}/txt_cnn.h5'.format(foldername))
		with open('{0}/tokenizer.pkl'.format(foldername), 'wb') as f:
			pkl.dump(self.T,f)

	def load(self, foldername='saved'):
		'''
		Load model from folder='foldername'
		'''
		self.cnn   = load_model('{0}/txt_cnn.h5'.format(foldername))
		with open('{0}/tokenizer.pkl'.format(foldername), 'rb') as f:
			self.T = pkl.load(f)

