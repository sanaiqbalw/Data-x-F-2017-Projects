import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
# use 1.0.0 to run
print(tf.__version__)

dir(tf.contrib)
# Load the data

# Load the data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
#lines = open('./movie_lines.txt').read().split('\n')
#conv_lines = open('./movie_conversations.txt').read().split('\n')

# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all of the conversations' lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

# Sort the sentences into questions (inputs) and answers2 (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# Check if we have loaded the data correctly
limit = 0
for i in range(limit+1000, limit+1005):
    print(questions[i])
    print(answers[i])
    print()

# Compare lengths of questions and answers2
print(len(questions))
print(len(answers))

# Take a look at some of the data to ensure that it has been cleaned well.
limit = 0
for i in range(limit+1000, limit+1005):
    print(questions[i])
    print(answers[i])
    print()

# Find the length of sentences
lengths = []
for question in questions:
    lengths.append(len(question.split()))
for answer in answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))

# Remove questions and answers2 that are shorter than 5 words and longer than 40 words.
min_line_length = 5
max_line_length = 40

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

i = 0

short_questions = []
short_answers = []

for i in range(len(questions)):
    if len(questions[i].split()) >= min_line_length and len(questions[i].split()) <= max_line_length and len(answers[i].split()) >= min_line_length and len(answers[i].split()) <= max_line_length:
        short_questions.append(questions[i])
        short_answers.append(answers[i])
# Filter out the answers2 that are too short/long



# Compare the number of lines we will use with the total number of lines.
print("# of questions:", len(short_questions))
print("# of answers2:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions)/len(questions),4)*100))


l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

def my_replace(raw_word):
    for j, term in enumerate(l1):
        raw_word = raw_word.replace(term,l2[j]).lower()
    for term in l3:
        raw_word = raw_word.replace(term,' ').lower()
    return raw_word



fq = open('context2', 'w')
fa = open('answers2', 'w')
for i in range(len(short_questions)):
    fq.write(my_replace(short_questions[i] + "\n"))
    fa.write(my_replace(short_answers[i] + "\n"))

fq.close()
fa.close()
exit(0)