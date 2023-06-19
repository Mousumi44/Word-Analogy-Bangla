from bnlp import BengaliGlove
import json
import sys
glove_path = "../Model/glove/bn_glove.39M.100d.txt"
bng = BengaliGlove()
embed_dict = bng.get_embed_dict(glove_path)


from my_bntransformer import MyBanglaTransformer
bnTransform  = MyBanglaTransformer()


def read_file():
    file1 = open('../Model/bnTransform_1024.txt', 'r')
    print(file1.readline())

def write_to_file():
    file1 = open("../Model/bnTransform_1024.txt", "a")
    i = 0
    for w in embed_dict.keys():
        try:
            e = bnTransform.get_word_embedding(w)
        except RuntimeError:
            print("Runtime Error")
            e = [0]*1024
        # e = e.tolist() #for laser, labse
        file1.write(w+" ")
        file1.write(str(e))
        file1.write("\n")
        if i%1000==0:
            print(f'word {i}')
        i+=1
    file1.close()

write_to_file()
# read_file()

#Laser
# from my_banglaBERT import MyBanglaBERT
# bnBERT = MyBanglaBERT()

# def read_file():
#     file1 = open('../Model/bnBERT_768.txt', 'r')
#     print(file1.readline())

# def write_to_file():
#     file1 = open("../Model/bnBERT_768.txt", "a")
#     i = 0
#     for w in embed_dict.keys():
#         try:
#             e = bnBERT.get_word_embedding(w)
#         except RuntimeError:
#             print("Runtime Error")
#             e = [0]*768
#         # e = e.tolist() #for laser, labse
#         file1.write(w+" ")
#         file1.write(str(e))
#         file1.write("\n")
#         if i%1000==0:
#             print(f'word {i}')
#         i+=1
#     file1.close()

# write_to_file()
# read_file()
        



# from sentence_transformers import SentenceTransformer, util
# import numpy as np

# # # Load the LaBSE model
# model_name = 'sentence-transformers/LaBSE'
# model = SentenceTransformer(model_name)


# # Define the Mikolov-style word analogy
# a = 'লম্বা'
# b = 'চারদিকে'
# c = 'দূরত্ব'

# # Tokenize the words
# a_tokens = model.tokenize(a)
# b_tokens = model.tokenize(b)
# c_tokens = model.tokenize(c)

# # Encode the words
# a_embedding = model.encode(a)
# b_embedding = model.encode(b)
# c_embedding = model.encode(c)

# print(a_embedding)
# # Compute the vector difference
# diff_vector = b_embedding - a_embedding + c_embedding

# # Find the closest word to the difference vector
# words = ['লম্বা', 'চারদিকে', 'দূরত্ব']
# word_embeddings = model.encode(words)
# similarity = util.pytorch_cos_sim(word_embeddings, diff_vector).cpu().detach().numpy().flatten()
# most_similar_word = words[np.argmax(similarity)]

# print(f'{a} is to {b} as {c} is to {most_similar_word}')

#Laser
# import numpy as np
# from laserembeddings import Laser

# # Load the pre-trained Laser model
# laser = Laser()

# # Define the Mikolov-style word analogy
# a = 'লম্বা'
# b = 'চারদিকে'
# c = 'দূরত্ব'

# # Tokenize the words
# a_tokens = a.split()
# b_tokens = b.split()
# c_tokens = c.split()

# # Encode the words
# a_embedding = laser.embed_sentences([a_tokens], lang='bn')[0]
# b_embedding = laser.embed_sentences([b_tokens], lang='bn')[0]
# c_embedding = laser.embed_sentences([c_tokens], lang='bn')[0]

# # Compute the vector difference
# diff_vector = b_embedding - a_embedding + c_embedding

# # Find the closest word to the difference vector
# words = ['লম্বা', 'চারদিকে', 'দূরত্ব']
# word_embeddings = laser.embed_sentences([words], lang='bn')[0]
# similarity = np.dot(word_embeddings, diff_vector) / (np.linalg.norm(word_embeddings) * np.linalg.norm(diff_vector))
# most_similar_word = words[np.argmax(similarity)]

# print(f'{a} is to {b} as {c} is to {most_similar_word}')

#BanglaBERT
# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np

# # Load the BanglaBERT model and tokenizer
# model_name = 'csebuetnlp/banglabert'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Define the Mikolov-style word analogy
# a = 'লম্বা'
# b = 'চারদিকে'
# c = 'দূরত্ব'

# # Tokenize the words
# a_tokens = tokenizer.tokenize(a)
# b_tokens = tokenizer.tokenize(b)
# c_tokens = tokenizer.tokenize(c)

# # Convert the tokens to IDs
# a_ids = tokenizer.convert_tokens_to_ids(a_tokens)
# b_ids = tokenizer.convert_tokens_to_ids(b_tokens)
# c_ids = tokenizer.convert_tokens_to_ids(c_tokens)

# # Convert the IDs to PyTorch tensors
# a_tensor = torch.tensor(a_ids).unsqueeze(0)
# b_tensor = torch.tensor(b_ids).unsqueeze(0)
# c_tensor = torch.tensor(c_ids).unsqueeze(0)

# # Encode the words
# with torch.no_grad():
#     a_embedding = model(a_tensor)[0].mean(1)
#     b_embedding = model(b_tensor)[0].mean(1)
#     c_embedding = model(c_tensor)[0].mean(1)

# # Compute the vector difference
# diff_vector = b_embedding - a_embedding + c_embedding

# # Find the closest word to the difference vector
# words = ['লম্বা', 'চারদিকে', 'দূরত্ব']
# word_ids = tokenizer.convert_tokens_to_ids(words)
# word_tensor = torch.tensor(word_ids).unsqueeze(0)
# with torch.no_grad():
#     word_embeddings = model(word_tensor)[0].mean(1)
# similarity = np.dot(word_embeddings, diff_vector.T) / (np.linalg.norm(word_embeddings, axis=1) * np.linalg.norm(diff_vector))
# most_similar_word = words[np.argmax(similarity)]

# print(f'{a} is to {b} as {c} is to {most_similar_word}')