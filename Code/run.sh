#!/bin/bash

datas=(Affix_বিভক্তি Tense Antonym_Adjective Antonym_Misc Comparative Division_District Gender Mikolov-Filtered Number_Date Number_Female Number_Ordinal Plural_Noun Plural_Object Prefix_উপসর্গ Suffix_নির্দেশক Suffix_প্রত্যয় Superlative সাধু_চলিত_conj সাধু_চলিত_noun সাধু_চলিত_pronoun সাধু_চলিত_verb)
directory="../Data"

for curFile in "${datas[@]}"
do
    echo $curFile
    # Run for curFile
    for f in "$directory"/*
    do
      if [[ -f "$f" ]] && [[ "$f" =~ "$curFile" ]]; then
        # echo "$f"
        python load_embeddings.py --embed="fasttext" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="glove" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="word2vec" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="laser" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="labse" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="bnBERT" --k=10 --file="$f" --curF=$curFile
        python load_embeddings.py --embed="bnTrans" --k=10 --file="$f" --curF=$curFile
        break
      fi
    done
done





