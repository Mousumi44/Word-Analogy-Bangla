from bnlp import BengaliWord2Vec, BengaliGlove
import argparse
import fasttext
import pandas as pd
from my_laser import BengaliLaser
from my_labse import BengaliLaBSE
from my_banglaBERT import MyBanglaBERT
from my_bntransformer import MyBanglaTransformer
import sys

# download the Bangla fasttext
# fasttext.util.download_model('bn', if_exists='ignore')

def fasttext_bn(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    # load the Bangla word vectors
    fast_model_path = "../Model/fasttext/cc.bn.300.bin"
    fast_bn = fasttext.load_model(fast_model_path)
    wordD_list = []
    for i in range(sample):
        wordD = fast_bn.get_analogies(wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return 

def BERT_bn(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    bert_path = "../Model/bnBERT_768.txt"
    bnbert = MyBanglaBERT()
    embed_dict = bnbert.get_embed_dict(bert_path)

    wordD_list = []
    for i in range(sample):
        wordD = bnbert.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return

def labse_bn(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    labse_path = "../Model/labse_768.txt"
    labseM = BengaliLaBSE()
    embed_dict = labseM.get_embed_dict(labse_path)

    wordD_list = []
    for i in range(sample):
        wordD = labseM.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return

def laser_bn(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    laser_path = "../Model/laser_1024.txt"
    laserM = BengaliLaser()
    embed_dict = laserM.get_embed_dict(laser_path)

    wordD_list = []
    for i in range(sample):
        wordD = laserM.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return


def glove_bn(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    # https://huggingface.co/sagorsarker/bangla-glove-vectors
    glove_path = "../Model/glove/bn_glove.39M.100d.txt"
    bng = BengaliGlove()
    embed_dict = bng.get_embed_dict(glove_path)
    

    wordD_list = []
    for i in range(sample):
        wordD = bng.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return


def word2vec_bn_wiki(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    # https://github.com/sagorbrur/bnlp
    bwv = BengaliWord2Vec()
    model_path = "../Model/word2vec/bnwiki_word2vec.model"
    vec = bwv.generate_word_vector(model_path, 'আমি')
    print(vec.shape)

    embed_dict = bwv.get_embed_dict(model_path)

    wordD_list = []
    for i in range(sample):
        wordD = bwv.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return

def bnTransformer(wordA_list, wordB_list, wordC_list, top):
    out_f_path = '../Result/'+args.embed+'/'+args.curF+'.txt'
    out_f = open(out_f_path, 'a')
    sample = len(wordA_list)
    bntrans_path = "../Model/bnTransform_1024.txt"
    bnTransform  = MyBanglaTransformer()
    embed_dict = bnTransform.get_embed_dict(bntrans_path)

    wordD_list = []
    for i in range(sample):
        wordD = bnTransform.get_analogies(embed_dict, wordA= wordA_list[i], wordB= wordB_list[i], wordC = wordC_list[i], k=top)
        wordD_list.append(wordD)
        print(f'{wordA_list[i]} {wordB_list[i]} {wordC_list[i]}')
        print(f'Predicted top {top} words: {wordD}')
        out_f.write(str(wordD))
        out_f.write('\n')
    return
def file_to_list(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f:
            words = line.strip().split()
            result.append(words)
    df = pd.DataFrame(result, columns=['wordA', 'wordB', 'wordC', 'wordD'])
    return df['wordA'].tolist(), df['wordB'].tolist(), df['wordC'].tolist(), df['wordD'].tolist()

def load_embed(args, wordA_list, wordB_list, wordC_list):
    if args.embed=="bnTrans":
        print(f'Model: {args.embed}')
        bnTransformer(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="bnBERT":
        print(f'Model: {args.embed}')
        BERT_bn(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="labse":
        print(f'Model: {args.embed}')
        labse_bn(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="laser":
        print(f'Model: {args.embed}')
        laser_bn(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="fasttext":
        print(f'Model: {args.embed}')
        fasttext_bn(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="word2vec":
        print(f'Model: {args.embed}')
        word2vec_bn_wiki(wordA_list, wordB_list, wordC_list, args.k)

    if args.embed=="glove":
        print(f'Model: {args.embed}')
        glove_bn(wordA_list, wordB_list, wordC_list, args.k)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", type=str, help="Embedding Type")
    parser.add_argument('--k', type=int, required=True, help="Top k similarities")
    parser.add_argument('--file', type=str, required=True, help="File Name")
    parser.add_argument('--curF', type=str, required=True, help=" Current Processed File")
    args = parser.parse_args()

    wordA_list, wordB_list, wordC_list, wordD_list = file_to_list(args.file)
    load_embed(args, wordA_list, wordB_list, wordC_list)