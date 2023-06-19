import argparse
import openai
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from load_embeddings import file_to_list

def create_response(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    api_key ="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",
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

def main(args):
    wordA_list, wordB_list, wordC_list, wordD_list = file_to_list(args.file)
    prompt = f"""I want you to act as a native Bangla linguist to test Mikolov-style word analogy.
            For a given {{relationship type}} in curly bracket and input wordA, wordB, wordC,  return
            wordD where wordD = wordB - wordA + wordC. Input format will
            be {{Relationship type}} wordA wordB wordC. Just give the list of top-10 words
            for wordD like a list of string, don't explain anything.
            Give the output in JSON format with key "wordD_list".
            My first input is 
            """
    prompt = prompt+' {'+args.rel+'} '+wordA_list[args.no]+' '+wordB_list[args.no]+' '+wordC_list[args.no]
    response = create_response(prompt)
    response = response.replace('\n', ' ')
    print(f'Prompt: {prompt}')
    print(f'Response: {response}')
    file1 = open("../Result/chatgpt/"+args.rel+".txt", "a")
    file1.write(f'Sample: {args.no} \n')
    file1.write(response+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rel', type=str, required=True, help="Relationship Type")
    parser.add_argument('--file', type=str, required=True, help="File Name")
    parser.add_argument('--no', type=int, help='sample no')
    args = parser.parse_args()
    main(args)