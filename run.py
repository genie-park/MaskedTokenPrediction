import torch
from transformers import BertTokenizer, BertForMaskedLM
from util import create_train_dataset, create_evaluate_dateset, save_checkpoint, get_candidate_vocab_mask
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import argparse
import os 
import numpy as np 
from model import MaskedModel

def train (model, tokenizer, args) :
    model.cuda()
    model.train()
    dataset = create_train_dataset(tokenizer, args.data_dir, args.max_seq_length, args.max_label_length)
    data_loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.train_batch_size)
    checkpoint = None 
    optimizer = optim.SGD(model.parameters(), lr=1e-5)    
    for epoch in range (0, args.n_epoch):
        for step, batch in enumerate(data_loader) :
            batch = [t.cuda() for t in batch]
            outputs = model(input_ids= batch[1], masked_lm_labels=batch[2])
            loss, prediction_scores = outputs[:2]
            
            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if step % 100 == 99:
                print( 'step: %d loss: %f' % (step, loss.item())) 
        checkpoint = save_checkpoint(model, args.output_dir, epoch)
        evaluate(model, tokenizer, args)               
    return checkpoint 


def evaluate(model, tokenizer, args):
    model.cuda()
    model.eval()
    dataset = create_evaluate_dateset(tokenizer, args.data_dir, args.max_seq_length, args.max_label_length)
    data_loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.eval_batch_size)
    candidate_mask = get_candidate_vocab_mask(args.data_dir, tokenizer.vocab_size)

    out_f = open(os.path.join(args.output_dir, 'result.txt'), 'w', encoding='utf8')
    hit = 0 
    trial = 0 
    with torch.no_grad(): 
        for step, batch in enumerate(data_loader):
            example_id = batch[0].numpy()
            input_ids = batch[1].cuda()
            label_ids = batch[2].numpy() 
            label_pos = [label_ids != -100]

            outputs = model(input_ids=input_ids)
            prediction_scores = outputs[0].detach().cpu().numpy() 
            prediction_scores[:, :, ~candidate_mask] = - np.inf            
            prediction = np.reshape(np.argmax(prediction_scores, axis=2)[label_pos], (-1, args.max_label_length))             
            label = np.reshape(label_ids[label_pos], (-1, args.max_label_length))

            trial += batch[0].shape[0]

            for index, item in enumerate(prediction):
                predicted = tokenizer.decode(prediction[index], skip_special_tokens=True)
                gt = tokenizer.decode(label[index], skip_special_tokens=True)
                if predicted == gt :
                    hit += 1
                out_f.write('%s\t %s\n' % (predicted, gt))

    print ('accuracy: %f hit: %d, trial: %d' % ((hit*100)/trial, hit, trial))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--n_epoch',type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=92)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--max_label_length', type=int, default=6)
    # parser.add_argument('--from_checkpoint', type=str, default='./output/19-33-48-0_model.bin' )
    parser.add_argument('--from_checkpoint', type=str)
    main_args = parser.parse_args()

    model = MaskedModel.from_pretrained('bert-base-uncased')        
    if main_args.from_checkpoint:         
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(main_args.from_checkpoint))        
    else:
        model = torch.nn.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if main_args.do_train : 
        checkpoint = train(model, tokenizer, main_args) 
        model.load_state_dict(torch.load(checkpoint))

    if main_args.do_eval : 
        evaluate(model, tokenizer, main_args) 
