from transformers import BertTokenizer

def compute_accuracy() :
    hit = 0
    trial = 0 
    with open ('./output/result.txt', 'r', encoding='utf8') as f:
        for line in f: 
            words = line.split()
            if len(words) == 2 and words[0] == words[1] :
                hit +=1
            trial += 1 
    print('accuracy: %d, hit:%d, trial:%d' % ((hit*100)/trial, hit, trial))

def split_training_csv():
    import random 
    train_f = open('./dataset/training.csv', 'w', encoding='utf8')
    dev_f = open('./dataset/dev.csv', 'w', encoding='utf8')

    with open('./dataset/original/training.csv', 'r', encoding='utf8') as f:
        header = f.readline() 
        for line in f:
            if random.randint(1, 10) < 2:
                dev_f.write(line)
            else:
                train_f.write(line) 

if __name__ == '__main__':
    # split_training_csv()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print( tokenizer.pad_token_id ) 