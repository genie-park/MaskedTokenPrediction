from pattern.en import conjugate, lemma
from collections import OrderedDict
import csv

exception = {
    'biding':'bid', 
    'born':'born',
    'drags':'drag',
    'favored':'favor','favoring':'favor',
    'focusing':'focus','focused':'focus',
    'halting':'halt', 'halts':'halt','halted':'halt',
    'hoping':'hope', 'hoped':'hope', 'hopes':'hope',
    'left':'leave',
    'ongoing':'ongoing',
    'paid':'pay',
    'pending':'pending',
    'putting':'put',
    'recovered':'recover',
    'signaling':'signal','signaled':'signal',
    'shrank':'shrink',
    'sped':'speed',
    'totaling':'total','totaled':'total',
    'waned':'wan',
    'waning':'wan'
}

def build_vocab():
    word_dict = {}
    missing = {}
    vocab = {} 

    with open('./dataset/HAAFOR_Challenge_2019_word_list.csv', 'r') as f:
        index = 0 
        for line in f:
            word = line.strip()
            word_dict[word] = index
            index += 1

    vocab_stats = {}
    with open ('./dataset/original/training.csv', 'r') as f:
        csv_reader = csv.reader(f) 
        next(csv_reader) 

        for line in csv_reader:
            word = line[2].strip()
            if vocab_stats.get(word) is not None:
                vocab_stats[word] += 1
            else:
                vocab_stats[word] = 1

            lemma_word = lemma(word)
            if exception.get(word) is not None:
                vocab[word] = word_dict[exception[word]]
                continue
            if missing.get(word) is not None:
                missing[word] +=1
                continue 
            if word_dict.get(lemma_word) is not None:
                vocab[word] = word_dict[lemma_word]
            else:
                missing[word] = 1 
    od = OrderedDict(sorted(vocab.items()))

    with open('./dataset/vocab.dic', 'w') as f :
        for word in od.keys():
            f.write('%s\t%d\n'%(word, od[word]))

    with open('./dataset/vocab.stats', 'w') as f :
        for word in vocab_stats.keys():
            f.write('%s\t%d\n'%(word, vocab_stats[word]))

def verify_vocab():
    vocab={}
    word_dict={}

    with open('./dataset/HAAFOR_Challenge_2019_word_list.csv', 'r') as f:
        index = 0 
        for line in f:
            word = line.strip()
            word_dict[index] = word
            index += 1
    
    with open('./dataset/vocab.dic', 'r') as f :
        for line in f:
            items = line.split()
            word = items[0]
            index = items[1]
            vocab[word] = int(index)
    
    with open('./dataset/original/training.csv', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        for line in csv_reader:
            wrod = line[2]

            if vocab.get(wrod) is None:
                print('vocab error' + word)

            if word_dict[vocab[word]] != lemma(word):
                if word_dict[vocab[word]] != exception[word]:
                    print('vocab error' + word)
    print('verify completed')    

if __name__ == '__main__':
    build_vocab()
    verify_vocab()



