from load_embeddings import file_to_list
import argparse
import ast
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def check_accuracy(result, predicted):
    if len(result) != len(predicted):
        print(f'Type: {args.curF} Embedding: {args.embed} Length Mismatch!!!')
    else:
        sample = len(result)
        match = 0
        for i in range(sample):
            rWord = result[i]
            predWord = predicted[i]
            if rWord in predWord:
                match +=1
        return format((match/(sample*1.0))*100, '.2f')

def read_output_file(filepath, top):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    result = []
    for line in lines:
        lst = ast.literal_eval(line)
        result.append(lst[:top])
    return result

def main(args):
    in_path = '../Data/'+args.curF+'.txt'
    if args.embed=="chatgpt" or args.embed=="bard":
        out_path = '../Result/'+args.embed+'/'+args.curF+'_v2.txt'
    else:
        out_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    _, _, _, wordD_list = file_to_list(in_path)
    pred_word_1 = read_output_file(out_path, top=1)
    pred_word_3 = read_output_file(out_path, top=3)
    pred_word_5 = read_output_file(out_path, top=5)
    pred_word_10 = read_output_file(out_path, top=10)
    acc_1 = check_accuracy(wordD_list, pred_word_1)
    acc_3 = check_accuracy(wordD_list, pred_word_3)
    acc_5 = check_accuracy(wordD_list, pred_word_5)
    acc_10 = check_accuracy(wordD_list, pred_word_10)
    # print(f'Type: {args.curF} Embedding: {args.embed} Top-{1}: {acc_1}% Top-{3}: {acc_3}% Top-{5}: {acc_5}% Top-{10}: {acc_10}%')
    print(f'{args.curF} {args.embed} {acc_1}% {acc_3}% {acc_5}% {acc_10}%')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", type=str, help="Embedding Type")
    parser.add_argument('--curF', type=str, required=True, help="input File for 4th word")
    args = parser.parse_args()
    main(args)