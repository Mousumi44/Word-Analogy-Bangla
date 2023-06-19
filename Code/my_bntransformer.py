from bntransformer import BanglaTokenizer
import torch
import transformers
import numpy as np
from scipy import spatial

class MyBanglaTransformer:
    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        self.model = transformers.BartModel.from_pretrained(model_name)

        # Set the model to evaluation mode
        self.model.eval()

        # Load the BanglaTokenizer
        self.tokenizer = BanglaTokenizer(model_path=model_name)

    def get_word_embedding(self, input_text):
        tokens = self.tokenizer.tokenize(input_text)

        # Convert the tokens to token IDs
        input_ids = self.tokenizer.encode(input_text)

        # Convert the input_ids to a PyTorch tensor
        input_tensor = torch.tensor(input_ids)

        # Generate the word embeddings
        with torch.no_grad():
            outputs = self.model(input_tensor.unsqueeze(0))
            embeddings = outputs.last_hidden_state.squeeze(0)

        embeddings = torch.mean(embeddings, dim=0)
        return embeddings.tolist()
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
    
    def get_embed_dict(self, bntrans_path):
        embeddings_dict = {}
        with open(bntrans_path, "r") as f: 
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
    word = "আমি"
    bnTransform  = MyBanglaTransformer()
    em = bnTransform.get_word_embedding(word)
    print(em)
    print(len(em))
