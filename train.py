#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler

from graph_sequence import *

def generate_model():
	model = Sequential([
		Dense(6, 
			input_shape=(14,),
			activation='tanh'),
		#Dense(6, activation='tanh'),
		Dense(1, activation='softmax'),
	])

	# For a single-input model with 10 classes (categorical classification):

	#model = Sequential()
	#model.add(Dense(32, activation='relu', input_dim=100))
	#model.add(Dense(10, activation='softmax'))
	#model.compile(optimizer='rmsprop',
	#	loss='categorical_crossentropy',
	#	metrics=['accuracy'])

	return model

def train(args):

	model = generate_model()
	model.compile(loss='mean_squared_error', #mean_squared_erro
				optimizer='adam', #adam
				metrics=['accuracy'])

	# Convert labels to categorical one-hot encoding
	seq_train = GraphSequence(args)
	#print(seq_train)
	
	model.fit_generator(seq_train, epochs=10)

	seq_test = GraphSequence(args, test=True)
	result = model.evaluate_generator(seq_test)

	print(f"Accuracy: {round(result[1]*100)}%")

	scalar = MinMaxScaler()
	# new instances where we do not know the answer
	Xnew, _ = make_blobs(n_samples=5, centers=None, n_features=14, random_state=1)
	scalar.fit(Xnew)
	Xnew = scalar.transform(Xnew)
	# make a prediction
	ynew = model.predict_classes(Xnew)
	# show the inputs and predicted outputs
	for i in range(len(Xnew)):
		print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, default="local")
	args = parser.parse_args()

	train(args)


