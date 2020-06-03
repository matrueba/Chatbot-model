import tensorflow as tf
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, rate, embedding_matrix=None):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        if embedding_matrix:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        #self.bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True))
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)
        self.dropout = tf.keras.layers.Dropout(rate=rate)

     
    def call(self, x):
        x = self.embedding(x)
        #x = self.bidirectional(x, initial_state = hidden)
        #output, state = self.dropout(x)
        output, state_h, state_c = self.lstm(x)    
        return output, state_h, state_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))
   


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, embedding_matrix=None):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        if embedding_matrix:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # Layers for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
    
        hidden_with_time_axis = tf.expand_dims(hidden[0], 1)
    
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, _, _ = self.lstm(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))

