import os
from torch.utils.data import TensorDataset
from transformers import BertTokenizer 
import torch
import csv 
from tqdm import tqdm
from datetime import datetime
import numpy as np 

def encode_context(tokenizer, context_text, label_word, max_seq_length, max_label_length, vocab):
    label = tokenizer.encode(label_word, add_special_tokens=False)
    label = [vocab[l] for l in label]
    label = label + [tokenizer.pad_token_id] * (max_label_length - len(label))

    context = [tokenizer.mask_token_id] * max_label_length    
    splited= context_text.split('[BLANK]')
    before_blank, after_blank = (splited[0], splited[1])
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

    vocab, _ = get_candidate_vocab_mapping(data_dir)
    for item in tqdm(csv_file, desc='Create dev data set '):        
        example_id = int(item[0])
        context = item[1]
        label = item[2] 

        example_ids.append(example_id)  
        context, label = encode_context(tokenizer, context, label, max_seq_length, max_label_length, vocab)
        labels.append(label)
        contexts.append(context)        

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

    vocab, _ = get_candidate_vocab_mapping(data_dir)

    for item in tqdm(csv_file, desc='Create training data set '):        
        example_id = int(item[0])
        context = item[1]
        label = item[2] 

        example_ids.append(example_id)  
        context, label = encode_context(tokenizer, context, label, max_seq_length, max_label_length, vocab)
        labels.append(label)
        contexts.append(context)        

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
    with open(os.path.join(data_dir, 'vocab_tokenid.dic'), 'r', encoding='utf8') as f:
        for line in f:
            token_id = int(line.strip())
            vocab[token_id] = True 
    return vocab        


def get_candidate_vocab_mapping(data_dir):
    vocab = {} 
    reverse_vocab = {}
    with open(os.path.join(data_dir, 'vocab_tokenid.dic'), 'r', encoding='utf8') as f:
        for index, line in enumerate(f):
            token_id = int(line.strip())
            vocab[token_id] = index
            reverse_vocab[index] = token_id
    return vocab, reverse_vocab


if __name__ == '__main__':
    create_train_dataset('./dataset/',256, 6)


    

