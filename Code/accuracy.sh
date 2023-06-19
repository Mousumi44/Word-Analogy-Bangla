# datas=(Affix_বিভক্তি Tense Antonym_Adjective Antonym_Misc Comparative Division_District Gender Mikolov-Filtered Division_District Gender Number_Date Number_Female Number_Ordinal Antonym_Adjective Antonym_Misc Affix_বিভক্তি Tense Comparative Plural_Noun Plural_Object Prefix_উপসর্গ Suffix_নির্দেশক Suffix_প্রত্যয় Superlative সাধু_চলিত_conj সাধু_চলিত_noun সাধু_চলিত_pronoun সাধু_চলিত_verb)

# embeds=(fasttext word2vec glove laser labse bnBERT bnTrans)

datas=(Comparative Prefix_উপসর্গ Suffix_নির্দেশক Suffix_প্রত্যয়)
embeds=(chatgpt)

for curFile in "${datas[@]}" 
do
    echo $curFile
    for embd in "${embeds[@]}"
    do
        python accuracy_check.py --embed=$embd --curF=$curFile 2>&1 | grep -v "This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags"
    done
done


