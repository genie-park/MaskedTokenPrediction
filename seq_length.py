import os 
import csv 
from tqdm import tqdm
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

fp = open(os.path.join('./dataset', 'training.csv'))
csv_file = csv.reader(fp)

trial = 0 
blank_num = 0 
seq_length = 0 
max_seq_length = 0 
for item in tqdm(csv_file, desc='Create training data set '):        
    trial += 1
    example_id = int(item[0])        
    context = item[1]
    label = item[2] 
    blocks = context.split('[BLANK]')
    token_length = len(tokenizer.encode(blocks[0])) + len(tokenizer.encode(blocks[1]))

    blank_num += (len(blocks) -1 )
    seq_length += token_length
    max_seq_length = max(max_seq_length, token_length)

print ('blank: %f seq_length: %f max: %d' % (blank_num/trial, seq_length/trial, max_seq_length))
                