import tensorflow as tf
import preprocess_data as pp
import chatbot_model as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import os


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def lstm_train(encoder, decoder, dataset, epochs, words2int, batch_size,  batches):

    optimizer = tf.train.AdamOptimizer()

    checkpoint_path = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    #train process
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:

                enc_output, enc_h, enc_c = encoder(inp)
                encoder_states = [enc_h, enc_c]
                dec_hidden = encoder_states
                dec_input = tf.expand_dims([words2int['<SOS>']] * batch_size, 1)       
                
                for t in range(1, targ.shape[1]):
                    
                    predictions, _ = decoder(dec_input, dec_hidden, enc_output)                   
                    loss += loss_function(targ[:, t], predictions)                  
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))           
            total_loss += batch_loss            
            variables = encoder.variables + decoder.variables         
            gradients = tape.gradient(loss, variables)           
            optimizer.apply_gradients(zip(gradients, variables))

            print_batch = 50
            if batch % print_batch == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
                print('Time taken for each {} batches {} sec'.format(print_batch, time.time() - start))

    #Save the model
    checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batches))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



def gru_train(encoder, decoder, dataset, epochs, words2int, batch_size, batches):

    optimizer = tf.train.AdamOptimizer()

    #train process
    for epoch in range(epochs):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:

                enc_output, enc_hidden,  = encoder(inp, hidden)
                dec_hidden = enc_hidden            
                dec_input = tf.expand_dims([words2int['<SOS>']] * batch_size, 1)       
                               
                for t in range(1, targ.shape[1]):
                    
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)                    
                    loss += loss_function(targ[:, t], predictions)                                      
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))            
            total_loss += batch_loss            
            variables = encoder.variables + decoder.variables           
            gradients = tape.gradient(loss, variables)           
            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batches))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



def evaluate(inputs, units, encoder, decoder, int2word, word2int, max_length):
    
    attention_plot = np.zeros((max_length, max_length))

    #Extract the original sentence using int2word
    sentence = ''
    for i in inputs[0]:
        if i == 0:
            break
        sentence = sentence + int2word[i] + ' '
    sentence = sentence[:-1]
    
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word2int['<SOS>']], 0)

    # start decoding
    for t in range(max_length): # limit the length of the decoded sequence
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += int2word[predicted_id] + ' '

        # stop decoding if '<EOS>' is predicted
        if int2word[predicted_id] == '<EOS>':
            return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention)

    ax.set_xticklabels([''] + sentence, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def predict_sentence(random_sentence, units, encoder, decoder, int2word, words2int, max_length):

    result, sentence, attention_plot = evaluate(random_sentence, units, encoder, decoder, int2word, words2int, max_length)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))





