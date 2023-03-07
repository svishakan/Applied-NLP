import csv
import json
from collections import Counter, defaultdict
import math
import numpy as np

def read_file(path):
    with open(path) as file:
        reader = list(csv.reader(file, delimiter="\t"))
        word_count = defaultdict(int)
        tag_count = defaultdict(int)
        for i, row in enumerate(reader):
            if row==[]:
                reader[i] = ['','','<start>']
                tag_count['<start>']+=1
            else:
                word_count[row[1]] += 1
                tag_count[row[2]] += 1
        for i, row in enumerate(reader):
            if word_count[row[1]]<3:
                word_count['<unk>']+=word_count[row[1]]
                del word_count[row[1]]
                reader[i][1] = '<unk>'
    return reader, word_count, tag_count

tagged_words, word_count, tag_count = read_file('data/train')
unknown_count = word_count['<unk>']
del word_count['<unk>']
word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
word_count.insert(0, ('<unk>', unknown_count))
word_list = list([i[0] for i in word_count])
tag_list = list(tag_count.keys())
tag_list.remove('<start>')

with open('vocab.txt', 'w') as f:
    for i,word in enumerate(word_count, start=1):
        f.write(f'{word[0]}\t{i}\t{word[1]}\n')

t_count = defaultdict(lambda: defaultdict(lambda:0.00001))
t_prob = defaultdict(lambda: defaultdict(lambda:0.00001))
e_count = defaultdict(lambda: defaultdict(lambda:0.00001))
e_prob = defaultdict(lambda: defaultdict(lambda:0.00001))

for i in range(len(tagged_words)-1):
    prev_tag = tagged_words[i][2]
    curr_tag = tagged_words[i+1][2]
    if prev_tag!='<start>':
        t_count[prev_tag][curr_tag] += 1

for prev_tag in t_count.keys():
    for curr_tag in t_count[prev_tag].keys():
            t_prob[prev_tag][curr_tag] = t_count[prev_tag][curr_tag]/tag_count[prev_tag]

for tagged_word in tagged_words:
    word=tagged_word[1]
    tag=tagged_word[2]
    if tag!='<start>':
        e_count[tag][word] += 1

for tag in e_count.keys():
    for word in e_count[tag].keys():
        e_prob[tag][word] = e_count[tag][word]/tag_count[tag]

t_prob_params = 0
e_prob_params = 0
for i in t_prob.keys():
    for j in t_prob[i].keys():
        t_prob_params += 1
        
for i in e_prob.keys():
    for j in e_prob[i].keys():
        e_prob_params += 1

print(len(word_list))
print(unknown_count)
print(t_prob_params)
print(e_prob_params)

hmm = {'transition':t_prob, 'emission': e_prob}
with open('hmm.json','w') as f:
    f.write(json.dumps(hmm))

def greedy_decode(hmm, orig_sequence, output_sentence, testing_flag=False):
    n = len(output_sentence)
    out_tag_seq = [0]*n
    index = np.argmax([hmm['transition']['start'][tag] * hmm['emission'][tag][output_sentence[0]] for tag in tag_list])
    out_tag_seq[0] = tag_list[index]
    for o in range(1,n):
        index = np.argmax([hmm['transition'][out_tag_seq[o-1]][tag] * hmm['emission'][tag][output_sentence[o]] for tag in tag_list])
        out_tag_seq[o] = tag_list[index]
    if testing_flag:
        with open('greedy.out', 'a') as f:
            for i in range(n):
                f.write(f'{i+1}\t{orig_sequence[i]}\t{out_tag_seq[i]}\n')
            f.write('\n')
    return out_tag_seq

with open('data/dev','r') as f:
    reader = list(csv.reader(f, delimiter="\t"))
    orig_words = []
    dev_words = []
    dev_tags = []
    pred_dev_tags = []
    for row in reader:
        if row==[]:
            pred_dev_tags.extend(greedy_decode(hmm, orig_words, dev_words))
            dev_words = []
            orig_words = []
        else:
            if row[1] in word_list:
                dev_words.append(row[1])
            else:
                dev_words.append('<unk>')
            orig_words.append(row[1])
            dev_tags.append(row[2])
    pred_dev_tags.extend(greedy_decode(hmm, orig_words, dev_words))
    correct_preds = 0
    for i in range(len(pred_dev_tags)):
        if pred_dev_tags[i] == dev_tags[i]:
            correct_preds+=1
    print(correct_preds/len(pred_dev_tags))

with open('data/test','r') as f:
    with open('greedy.out','w') as f2:
        pass
    reader = list(csv.reader(f, delimiter="\t"))
    orig_words = []
    test_words = []
    pred_test_tags = []
    for row in reader:
        if row==[]:
            pred_test_tags.extend(greedy_decode(hmm, orig_words, test_words, True))
            test_words = []
            orig_words = []
        else:
            if row[1] in word_list:
                test_words.append(row[1])
            else:
                test_words.append('<unk>')
            orig_words.append(row[1])
    pred_test_tags.extend(greedy_decode(hmm, orig_words, test_words, True))

def viterbi_decode(hmm, orig_sequence, output_sequence, testing_flag = False):
    V = [{}]
    for s in tag_list:
        V[0][s] = {'prob': hmm['transition']['<start>'][s] * hmm['emission'][s][output_sequence[0]], 'bp': '<start>'}
    for o in range(1, len(output_sequence)):
        V.append({})
        for s in tag_list:
            max_prob = max([(prev_st, V[o-1][prev_st]['prob'] * hmm['transition'][prev_st][s] * hmm['emission'][s][output_sequence[o]]) for prev_st in tag_list], key=lambda x:x[1])
            V[o][s] = {'prob': max_prob[1], 'bp': max_prob[0]}
    path = []

    max_prob = max([(s, V[-1][s]['prob']) for s in tag_list], key=lambda x:x[1])
    path.append(max_prob[0])
    prev_st = max_prob[0]

    for o in range(len(V)-2, -1, -1):
        val = V[o+1][prev_st]['bp']
        if val!='<start>':
            path.insert(0, V[o+1][prev_st]['bp'])
        prev_st = V[o+1][prev_st]['bp']

    if testing_flag:
        with open('viterbi.out', 'a') as f:
            for i in range(len(output_sequence)):
                f.write(f'{i+1}\t{orig_sequence[i]}\t{path[i]}\n')
            f.write('\n')
        
    return path

with open('data/dev','r') as f:
    reader = list(csv.reader(f, delimiter="\t"))
    dev_words = []
    dev_tags = []
    pred_dev_tags = []
    orig_words = []
    for row in reader:
        if row==[]:
            pred_dev_tags.extend(viterbi_decode(hmm, orig_words, dev_words))
            dev_words = []
            orig_words = []
        else:
            if row[1] in word_list:
                dev_words.append(row[1])
            else:
                dev_words.append('<unk>')
            orig_words.append(row[1])
            dev_tags.append(row[2])
    pred_dev_tags.extend(viterbi_decode(hmm, orig_words, dev_words))
    correct_preds = 0
    for i in range(len(pred_dev_tags)):
        if pred_dev_tags[i] == dev_tags[i]:
            correct_preds+=1
    print(correct_preds/len(pred_dev_tags))

with open('data/test','r') as f:
    with open('viterbi.out','w') as f2:
            pass
    reader = list(csv.reader(f, delimiter="\t"))
    test_words = []
    pred_test_tags = []
    orig_words = []
    for row in reader:
        if row==[]:
            pred_test_tags.extend(viterbi_decode(hmm, orig_words, test_words, True))
            test_words = []
            orig_words = []
        else:
            if row[1] in word_list:
                test_words.append(row[1])
            else:
                test_words.append('<unk>')
            orig_words.append(row[1])
    pred_test_tags.extend(viterbi_decode(hmm, orig_words, test_words, True))