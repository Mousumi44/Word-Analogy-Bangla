
import argparse
from load_embeddings import file_to_list
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bard_api.bard import get_response

def main(args):
    wordA_list, wordB_list, wordC_list, wordD_list = file_to_list(args.file)
    prompt = f"""I want you to act as a native Bangla linguist to test Mikolov-style word analogy.
            For a given {{relationship_type}}in curly brackets and input wordA, wordB, wordC, return 
            wordD where wordD = wordB - wordA + wordC. 
            Input format will be {{relationship_type}} wordA wordB wordC. 
            Just give the list of top-10 words for wordD like a list of strings, 
            don't explain anything, just give the output in JSON format with key "wordD_list".
            First input is
            """
    input = '{'+args.rel+'} '+wordA_list[args.no]+' '+wordB_list[args.no]+' '+wordC_list[args.no]
    prompt = prompt+' '+input
    response = get_response(prompt)

    response = response.replace(input, ' ')
    response = response.replace('\n', ' ')
    print(f'Prompt: {prompt}')
    print(f'Response: {response}')
    file1 = open("../Result/bard/"+args.rel+".txt", "a")
    file1.write(f'Sample: {args.no} \n')
    file1.write(response+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rel', type=str, required=True, help="Relationship Type")
    parser.add_argument('--file', type=str, required=True, help="File Name")
    parser.add_argument('--no', type=int, help='sample no')
    args = parser.parse_args()
    main(args)

