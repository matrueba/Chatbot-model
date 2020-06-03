import numpy as np
import re
import tensorflow as tf

#Create the dictionay that maps each sentence with its ID
def lines_dict(lines):
    id2line = {}
    for line in lines:
        _line = line.split(" +++$+++ ")
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


#Extract the conversations and set in a list
def conversations_extractor(conversations):
    conversations_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
        conversations_ids.append(_conversation.split(","))
    return conversations_ids

#Create the dataset with the question as features and the answers as labels
def question_answers_creator(conversations_ids, id2line):
    questions = []
    answers = []
    for conversation in conversations_ids:
        for i in range (len(conversation) - 1):

            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i+1]])
    return questions, answers

def clean_with_regular(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she es", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=|.?,!']", "", text)
    return text

#Clean the text
def clean_text(questions, answers):
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_with_regular(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_with_regular(answer))
    return clean_questions, clean_answers


#Create a dictionary that maps each word with the number of appearances
def count_words(clean_questions, clean_answers):
    word2count = {}

    for question in clean_questions:
        for word in question.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    for answer in clean_answers:
        for word in answer.split():
            if word not in word2count:
                word2count[word] = 1
            else:
                word2count[word] += 1

    return word2count


#Create a dictionary that maps each word with and unique ID
def words_to_int_dict(word2count):
    threshold = 20
    word_number = 1
    words2int = {}
    for word, count in word2count.items():
        if count >= threshold:
            words2int[word] = word_number
            word_number += 1

    return words2int


# Add tokens to dictionary
def add_tokens(words2int):

    tokens = ['<EOS>', '<OUT>', '<SOS>']
    words2int['<PAD>'] = 0

    for token in tokens:
        words2int[token] = len(words2int) + 1
    
    return words2int


# Add end of sentence(EOS) to each answer
def add_start_end_tokens(clean_questions, clean_answers):
    for i in range(len(clean_answers)):
        clean_answers[i] += ' <EOS>'
        clean_answers[i] = '<SOS> ' + clean_answers[i]
    for i in range(len(clean_questions)):
        clean_questions[i] += ' <EOS>'
        clean_questions[i] = '<SOS> ' + clean_questions[i] 
    return clean_questions, clean_answers


#Translate each word of each sentence to a int number
def translate_to_int(clean_answers, clean_questions, words2int):

    questiosns_into_int = []
    for question in clean_questions:
        ints = []
        for word in question.split():
            if word not in words2int:
                ints.append(words2int['<OUT>'])
            else:
                ints.append(words2int[word])
        questiosns_into_int.append(ints)

    answers_into_int = []
    for answer in clean_answers:
        ints = []
        for word in answer.split():
            if word not in words2int:
                ints.append(words2int['<OUT>'])
            else:
                ints.append(words2int[word])
        answers_into_int.append(ints)

    return questiosns_into_int, answers_into_int

def delete_long_sentences(questions, answers, max_len):
    short_questions = []
    short_answers = []
    for ques, ans in zip(questions, answers):
        if len(ques) <= max_len and len(ans) <= max_len:
            short_questions.append(ques)
            short_answers.append(ans)

    return short_questions, short_answers

def apply_padding(sequences):
    max_sequence_length = max([len(sequence) for sequence in sequences])
    final_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return final_sequences

#Complete preprocess
def preprocess_dataset(lines, conversations):
    id2line = lines_dict(lines)
    conversations_ids = conversations_extractor(conversations)
    questions, answers = question_answers_creator(conversations_ids, id2line)
    clean_questions, clean_answers = clean_text(questions, answers)
    word2count = count_words(clean_questions, clean_answers)
    words2int = words_to_int_dict(word2count)
    words2int = add_tokens(words2int)
    int2word = {w_i:w for w, w_i in words2int.items()}
    clean_questions, clean_answers = add_start_end_tokens(clean_questions, clean_answers)
    questiosns_into_int, answers_into_int = translate_to_int(clean_answers, clean_questions, words2int)
    questiosns_into_int, answers_into_int = delete_long_sentences(questiosns_into_int, answers_into_int, max_len=30)
    padded_questions = apply_padding(questiosns_into_int)
    padded_answers = apply_padding(answers_into_int)
    return padded_questions, padded_answers, words2int, int2word
