from word_forms.word_forms import get_word_forms
from transformers import BertTokenizer
from collections import OrderedDict
import csv 

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

word_dict = {}
with open('./dataset/HAAFOR_Challenge_2019_word_list.csv') as f:
    for line in f:
        word = line.strip()
        word_dict[word] = 1 

        words = get_word_forms(word)['v']
        for word in words :
            word_dict[word] = 1 

with open('./dataset/original/training.csv') as f:
    for line in csv.reader(f):        
        if word_dict.get(line[2]) is None:
            print ('missing word: %s token:%s ' % (line[2], tokenizer.encode(line[2], add_special_tokens=False)))

token_id_dict = {}
with open('./dataset/vocab.dic', 'w', encoding='utf8') as f: 
    for word in word_dict.keys():
        ids = tokenizer.encode(word, add_special_tokens=False)
        for id in ids:
            if token_id_dict.get(id) is None :
                token_id_dict[id] = 1 
            else: 
                token_id_dict[id] += 1 
        f.write('%s \t %s\n' % (word, ' '.join([str(id) for id in ids])))

ordered_d1 = OrderedDict(sorted(token_id_dict.items(), key=lambda t:t[0]))
mapping = []

with open('./dataset/vocab_tokenid.dic', 'w', encoding='utf8') as f: 
    for index, token_id in enumerate(ordered_d1):
        f.write(str(token_id) + '\n')
        mapping.append(token_id)

# print(len(mapping))

# for id in mapping :
#     print (id) 