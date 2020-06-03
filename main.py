import tensorflow as tf
from preprocess_data import *
import numpy as np
from train_test_chatbot import *
from sklearn.model_selection import train_test_split


print("Tensorflow version: " + str(tf.version.VERSION))
tf.compat.v1.enable_eager_execution()

#Load the files with conversations 
lines = open("data/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("data/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#Complete preprocess method
padded_questions, padded_answers, words2int, int2word = preprocess_dataset(lines, conversations)

#Process to generate te inputs and targets of the model
train_inputs, test_inputs, train_targets, test_targets = train_test_split(padded_questions,padded_answers, test_size=0.2)

print(train_inputs.shape)
print(train_targets.shape)

embedding_dim = 256
units = 64
rate=0.5
epochs = 10
vocab_size = len(words2int) + 1 
BATCH_SIZE = 32
BUFFER_SIZE = len(padded_questions)
N_BATCH = BUFFER_SIZE//BATCH_SIZE
steps_per_epoch = len(padded_questions)//BATCH_SIZE
max_length = train_inputs.shape[1]

#From the inputs and targets, create a batch generator to iterate over it
dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).shuffle(BUFFER_SIZE)
#Establish the batch size to iterate
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#Print a bach dimensions example
example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape)
print(example_target_batch.shape)

#Initialize encoder and decoder
encoder = cm.Encoder(vocab_size, embedding_dim, units, BATCH_SIZE, rate)
decoder = cm.Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

#Train the model
lstm_train(encoder, decoder, dataset, epochs, words2int, BATCH_SIZE, N_BATCH)

#Select random sentence to check the model
random_sentence = test_inputs[np.random.randint(0, len(test_inputs))]
random_sentence = np.expand_dims(random_sentence,0)

#Make a prediction with a random sentence of test set an plot confussion matrix
predict_sentence(random_sentence, units, encoder, decoder, int2word, words2int, max_length)



