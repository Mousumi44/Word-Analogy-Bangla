if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_sample> $1 <num_iterations>"
    exit 1
fi

# Set the starting sample no and number of iterations
start=$1
num_iterations=$2

datas=(Affix_বিভক্তি Tense Antonym_Adjective Antonym_Misc Comparative Division_District Gender Mikolov-Filtered Number_Date Number_Female Number_Ordinal Plural_Noun Plural_Object Prefix_উপসর্গ Suffix_নির্দেশক Suffix_প্রত্যয় Superlative সাধু_চলিত_conj সাধু_চলিত_noun সাধু_চলিত_pronoun সাধু_চলিত_verb)
directory="../Data"

for curFile in "${datas[@]}"
do
    # echo "$curFile"
    for f in "$directory"/*
    do
        if [[ -f "$f" ]] && [[ "$f" =~ "$curFile" ]]; then
            # echo "$f"
            for (( i=$start; i<$start+$num_iterations; i++ )); do
                python util_bard.py --rel="$curFile" --file="$f" --no=$i
                sleep 10
            done
        break
        fi
    done
done