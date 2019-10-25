#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing

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

	#helpers.execute_on_pipeline(ohe_pipeline, X_train, y_train, X_test, y_test)
	model.compile(loss='mean_squared_error', #mean_squared_erro
				optimizer='adam', #adam
				metrics=['accuracy'])

	# Convert labels to categorical one-hot encoding
	#one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
	seq_train = GraphSequence(args)
	#print(seq_train)
	
	model.fit_generator(seq_train, epochs=10)

	seq_test = GraphSequence(args, test=True)
	result = model.evaluate_generator(seq_test)

	print(f"Accuracy: {round(result[1]*100)}%")


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--database', type=str, default="local")
	args = parser.parse_args()

	train(args)


