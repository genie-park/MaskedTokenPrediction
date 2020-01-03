import os
from torch.utils.data import TensorDataset
from transformers import BertTokenizer 
import torch
import csv 
from tqdm import tqdm
from datetime import datetime
import numpy as np 

def encode_context(tokenizer, before_blank, after_blank, label_word, max_seq_length, max_label_length):
    label = tokenizer.encode(label_word, add_special_tokens=False)
    label = label + [tokenizer.pad_token_id] * (max_label_length - len(label))

    context = [tokenizer.mask_token_id] * max_label_length    

    before_word_count, after_word_count = (0, 0)    
    before_blank = tokenizer.encode(before_blank, add_special_tokens=False,)
    after_blank = tokenizer.encode(after_blank, add_special_tokens=False) 

    if len(before_blank) + len(after_blank) < max_seq_length - max_label_length - 2:
        context = [tokenizer.cls_token_id] + before_blank + context + after_blank 
        context = context + [tokenizer.pad_token_id] * (max_seq_length - len(context) - 1) + [tokenizer.sep_token_id]
        label = [-100] + [-100] * len(before_blank) + label + [-100] * len(after_blank) 
        label = label + [-100] * (max_seq_length - len(label) - 1) + [-100]
    else:
        while len(context) < max_seq_length - 2 : 
            if len(after_blank) == 0  or ( len(before_blank) != 0 and before_word_count < after_word_count) :
                before_word_id = before_blank.pop()
                context.insert(0, before_word_id )
                label.insert(0, -100)
                before_word_count +=1 
            elif len(after_blank) != 0 :
                next_word_id = after_blank.pop(0)
                context.append(next_word_id)
                label.append(-100)
                after_word_count += 1             
            else:
                print('tokenizer Error') 
                exit()                

        context = [tokenizer.cls_token_id] + context + [tokenizer.sep_token_id]
        label = [-100] + label + [-100]
    
    assert( len(context) == max_seq_length )
    assert( len(label) == max_seq_length )
    return context , label


def create_evaluate_dateset(tokenizer, data_dir, max_seq_length, max_label_length, has_header=False):
    cached_path = os.path.join(data_dir, '__cache_dev_dataset')
    if os.path.exists(cached_path):
        return torch.load(cached_path)      

    fp = open(os.path.join(data_dir, 'dev.csv'))
    csv_file = csv.reader(fp)
    if has_header : 
        header = next(csv_file)
        print('csv file header' + str(header))

    example_ids = []
    contexts = [] 
    labels = [] 
    label_masks = [] 

    for item in tqdm(csv_file, desc='Create dev data set '):        
        example_id = int(item[0])
        context = item[1]
        label = item[2] 
        
        blocks = context.split('[BLANK]')
        before_blank = blocks[0]
        after_blank = blocks[1]
        context_ids, label_ids = encode_context(tokenizer, before_blank, after_blank, label, max_seq_length, max_label_length)
        example_ids.append(example_id)  
        labels.append(label_ids)
        contexts.append(context_ids)     

    dataset = TensorDataset( torch.tensor(example_ids, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long))

    torch.save(dataset, cached_path)
    return dataset


def create_train_dataset(tokenizer, data_dir, max_seq_length, max_label_length, has_header=False):
    cached_path = os.path.join(data_dir, '__cache_train_dataset')
    if os.path.exists(cached_path):
        return torch.load(cached_path)
    
    fp = open(os.path.join(data_dir, 'training.csv'))
    csv_file = csv.reader(fp)
    if has_header : 
        header = next(csv_file)
        print('csv file header' + str(header))

    example_ids = []
    contexts = [] 
    labels = [] 
    label_masks = [] 
    vocab = get_vocab(data_dir)
    f = open(os.path.join(data_dir, 'aug_train.csv'), 'w', encoding='utf8')
    for item in tqdm(csv_file, desc='Create training data set '):        
        example_id = int(item[0])
        context = item[1]
        label = item[2] 
        blocks = context.split('[BLANK]')
        for index, before_blank in enumerate(blocks):
            if index < len(blocks) - 1: 
                after_blank = blocks[index+1]
                context_ids, label_ids = encode_context(tokenizer, before_blank, after_blank, label, max_seq_length, max_label_length)
                f.write('%d, %s[BLANK]%s, %s\n' % (example_id, before_blank, after_blank, label))
                example_ids.append(example_id)  
                labels.append(label_ids)
                contexts.append(context_ids)

        context = context.replace('[BLANK]', label)
        blocks = context.split()        
        for index, label in enumerate(blocks):
            if vocab.get(label) is not None and 3 < index < len(blocks) - 3 :
                before_blank = ' '.join(blocks[:index])
                after_blank = ' '.join(blocks[index+1:])
                context_ids, label_ids = encode_context(tokenizer, before_blank, after_blank, label, max_seq_length, max_label_length)
                f.write('%d, %s[BLANK]%s, %s\n' % (example_id, before_blank, after_blank, label))                
                example_ids.append(example_id)  
                labels.append(label_ids)
                contexts.append(context_ids)
    f.close()
    dataset = TensorDataset( torch.tensor(example_ids, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long))
    torch.save(dataset, cached_path)
    return dataset


def save_checkpoint(model, data_dir, epoch) :
    save_path = os.path.join(data_dir, datetime.now().strftime('%H-%M-%S') + '-' + str(epoch) + '_model.bin')
    torch.save(model.state_dict(), save_path)
    return save_path 

def get_candidate_vocab_mask(data_dir, vocab_size):
    vocab = np.zeros(vocab_size, dtype=bool) 
    with open(os.path.join(data_dir, 'vocab.dic'), 'r', encoding='utf8') as f:
        for line in f:
            token_id = int(line.strip())
            vocab[token_id] = True 
    return vocab        

def candidate_search(scores, vocab_mask, dependency_mask):
    predicted = [] 
    mask = vocab_mask 

    for token_score in scores:
        token_score[~mask] = -np.inf
        candidate_token_id = np.argmax(token_score)
        if candidate_token_id == 0:
            break 
        predicted.append(candidate_token_id)
        mask = dependency_mask(candidate_token_id)
    return predicted 

def get_vocab_dependency(data_dir, vocab_size):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    
    dependency = {}
    first_token_mask = np.zeros(vocab_size, dtype=bool)

    with open(os.path.join(data_dir, 'vocab.dic'), 'r', encoding='utf8') as f:
        for line in f:
            words = line.strip().split()            
            token_id = tokenizer.encode(words[0], add_special_tokens=False)
            first_token_mask[token_id[0]] = True
            for i in range(0, len(token_id) - 1):
                if dependency.get(token_id[i]) is None:
                    dependency[token_id[i]] = set([token_id[i+1]])
                else:
                    dependency[token_id[i]].add(token_id[i+1])

            if dependency.get(token_id[-1]) is None:
                dependency[token_id[-1]] = set([0])
            else:
                dependency[token_id[-1]].add(token_id[-1])

    for key, values in dependency.items():
        mask = np.zeros(vocab_size, dtype=bool)
        for id in values:
            mask[id] = True
        dependency[key] = mask 
    return first_token_mask, dependency

def get_candidate_vocab_mapping(data_dir):
    vocab = {} 
    reverse_vocab = {}
    with open(os.path.join(data_dir, 'vocab_tokenid.dic'), 'r', encoding='utf8') as f:
        for index, line in enumerate(f):
            token_id = int(line.strip())
            vocab[token_id] = index
            reverse_vocab[index] = token_id
    return vocab, reverse_vocab

def build_vocab(data_dir):
    vocab = {}
    missing_vocab = {} 
    from word_forms.word_forms import get_word_forms    
    with open(os.path.join(data_dir, 'HAAFOR_Challenge_2019_word_list.csv'), 'r', encoding='utf8') as f: 
        for word in f : 
            conjugates = get_word_forms(word.strip())['v']
            for verb in conjugates: 
                vocab[verb] = 0

    with open(os.path.join(data_dir, 'original', 'training.csv'), 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) 
        for data in csv_reader:
            if vocab.get(data[2]) is None:
                missing_vocab[data[2]] = 1 
            else:
                vocab[data[2]] += 1

    with open(os.path.join(data_dir, 'vocab.dic'), 'w', encoding='utf8') as f: 
        for verb in vocab: 
            if verb is not None: 
                f.write(verb + '\t' + str(vocab[verb]) +'\n')
        for verb in missing_vocab:
            f.write(verb + '\t missing\n')

def verify_vocab(data_dir) :
    vocab={} 
    with open(os.path.join(data_dir, 'vocab.dic'), 'r', encoding='utf8') as f: 
        for line in f: 
            data = line.split()
            vocab[data[0]] = 0

    missing_vocab = {} 
    with open(os.path.join(data_dir, 'original', 'training.csv'), 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) 
        for data in csv_reader:
            if vocab.get(data[2]) is None:
                missing_vocab[data[2]] = 1 
            else:
                vocab[data[2]] += 1

    with open(os.path.join(data_dir, 'vocab.dic_v'), 'w', encoding='utf8') as f: 
        for verb in vocab: 
            if verb is not None: 
                f.write(verb + '\t' + str(vocab[verb]) +'\n')
        for verb in missing_vocab:
            f.write(verb + '\t missing\n')
            
def get_vocab(data_dir) : 
    vocab={} 
    with open(os.path.join(data_dir, 'vocab.dic'), 'r', encoding='utf8') as f: 
        for line in f: 
            data = line.split()
            vocab[data[0]] = 0
    return vocab 


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    data_dir = './dataset'
    create_train_dataset(tokenizer, data_dir, 256, 6)
    # verify_vocab(data_dir) 


    

