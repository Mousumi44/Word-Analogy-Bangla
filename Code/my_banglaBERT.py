from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy import spatial

class MyBanglaBERT:
    def __init__(self):
        model_name = 'csebuetnlp/banglabert'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    def preprocess_str_list(self, string_list):
        # Combine the strings in the list into a single string
        combined_string = ''.join(string_list)

        # Remove the opening and closing square brackets from the string
        trimmed_string = combined_string.strip('[]')

        # Split the string into a list of strings, where each string represents a float value
        float_string_list = trimmed_string.split(',')

        # Convert each string in the list to a float and store it in a new list
        float_list = [float(val) for val in float_string_list]
        return float_list

    def get_word_embedding(self, word):
        word_tokens = self.tokenizer.tokenize(word)
        word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
        # Convert the IDs to PyTorch tensors
        word_tensor = torch.tensor(word_ids).unsqueeze(0)
        with torch.no_grad():
            word_embedding = self.model(word_tensor)[0].mean(1)
        return word_embedding.tolist()[0]
    
    def get_embed_dict(self, bnbert_path):
        embeddings_dict = {}
        with open(bnbert_path, "r") as f: 
            for line in f:
                values = line.split()
                word = values[0]
                vector = self.preprocess_str_list(values[1:])
                embeddings_dict[word] = np.array(vector)
        return embeddings_dict
    
    def get_analogies(self,embeddings_dict, wordA, wordB, wordC, k=10):
        def find_closest_embeddings(embedding):
            return sorted(
                embeddings_dict.keys(),
                key=lambda word: spatial.distance.euclidean(
                    embeddings_dict[word], embedding
                ),
            )
        try: 
            vecA = embeddings_dict[wordA]
            vecB = embeddings_dict[wordB]
            vecC = embeddings_dict[wordC]
            vecAB = vecB - vecA + vecC
            result = find_closest_embeddings(vecAB)[:k]
        except KeyError:
            print("Key not found")
            result = ['' for _ in range(k)]
        return result
    
if __name__=='__main__':
    mybnbert = MyBanglaBERT()
    # e = mybnbert.get_word_embedding(word='চারদিকে')
    # print(e)
    # print(len(e))
    embed_dict = mybnbert.get_embed_dict("../Model/bnBERT_768.txt")
    print(embed_dict['চারদিকে'])