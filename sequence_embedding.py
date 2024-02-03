import json
import torch
from torch import Tensor
import torch.nn as nn
from tqdm.notebook import tqdm
import os
import re
import pickle
import numpy as np
from gensim.models import Word2Vec
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

# model = Word2Vec.load("models/word2vec_model_random_walks_gcj.bin")
# attention_model = SelfAttentionSequenceEmbedding(input_size=100, num_heads=5).to(device)
word_model = Word2Vec.load("models/word2vec_model_bcb.bin")

node_type_dict = {'MethodDeclaration': 0, 'FormalParameter': 1, 'LocalVariableDeclaration': 2, 'VariableDeclarator': 3,
                  'BinaryOperation': 4, 'IfStatement': 5, 'BlockStatement': 6, 'StatementExpression': 7,
                  'Assignment': 8,
                  'MethodInvocation': 9, 'ForStatement': 10, 'TryStatement': 11, 'ClassCreator': 12, 'CatchClause': 13,
                  'WhileStatement': 14, 'ReturnStatement': 15, 'OtherStmt': 16}

node_index_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0,
                   '5': 0, '6': 0, '7': 0, '8': 0,
                   '9': 0,
                   '10': 0, '11': 0, '12': 0, '13': 0,
                   '14': 0,
                   '15': 0, '16': 0}

stmt_list = node_type_dict.keys()


def find_root_name(list):
    root_name = None
    for i in list:
        if i in stmt_list:
            root_name = i
            break

    if root_name is None:
        root_name = 'OtherStmt'

    node_index_dict[str(node_type_dict[root_name])] += 1
    return root_name


def get_stmt_sequence(node_sequence):
    stmt_index = []
    word_bag = []

    for i in range(len(node_sequence)):
        if node_sequence[i] in stmt_list:
            stmt_index.append(i)

    pairs = [[stmt_index[i], stmt_index[i + 1]] for i in range(len(stmt_index) - 1)]
    if len(stmt_index) >= 2:
        if pairs[-1][1] != len(node_sequence):
            pairs.append([pairs[-1][1], len(node_sequence)])
        for i in range(len(pairs)):
            if pairs[i][1] - pairs[i][0] <= 2 and i != len(pairs) - 1:
                pairs[i + 1][0] = pairs[i][0]
            else:
                word_bag.append(node_sequence[pairs[i][0]:pairs[i][1]])
    else:
        word_bag.append(node_sequence)

    return word_bag


def get_embedding(node, nodes, node_sequence):
    sequence_index = nodes.index(node)
    word = node_sequence[sequence_index]
    node_embedding = model.wv[str(node)]
    word_embedding = word_model.wv[word]
    #     embed = np.concatenate([word_embedding, node_embedding])
    embed = word_embedding * node_embedding
    return embed


def compute_Embedding(path):
    with open(path, 'rb') as file_1:
        node_dict = pickle.load(file_1)

    embeddings = torch.zeros(17, 100).to(device)

    for k, v in node_dict.items():
        G = v[0]
        node_sequence = v[1]
        centrality = v[2]  # 该社区的中心性
        nodes = list(G.nodes())
        word_bag = get_stmt_sequence(node_sequence)
        start_index = 0
        for sequence in word_bag:
            root_name = find_root_name(sequence)  # 子树根节点的名称

            index = node_type_dict[root_name]  # 在嵌入维度中的行索引

            node_index = nodes[start_index:start_index + len(sequence)]

            word_vectors = []

            #             embedding = np.mean([get_embedding(node, nodes, node_sequence) for node in node_index], axis=0)
            embedding = np.mean([word_model.wv[word] for word in sequence], axis=0)
            #             embedding = np.mean([model.wv[str(node)] for node in node_index], axis=0)

            embedding = torch.tensor(embedding, device=device)

            #             embedding = embedding * centrality

            embeddings[index] = embeddings[index] + embedding

            start_index += len(sequence)

    return embeddings


# def main():
#     for s in range(12):
#         input_path = "processed_data/GCJ/GCJ_json/googlejam4_src/" + str(s+1) + '/'
#     #     output_path = "processed_data/embeddings/"
#         output_path = "processed_data/GCJ/embeddings/googlejam4_src/" + str(s+1) + '/'

#         files = os.listdir(input_path)

#         for index, file in tqdm(enumerate(files), total=len(files), desc="Processing"):
#             embeddings = compute_Embedding(input_path + file)
#             torch.save(embeddings, output_path + file + '.pt')

#         print(i)


def main():
    input_path = "processed_data/node_pkl_subgraph/"
    output_path = "processed_data/embeddings/"

    files = os.listdir(input_path)

    print(len(files))

    for index, file in tqdm(enumerate(files), total=len(files), desc="Processing"):
        embeddings = compute_Embedding(input_path + file)
        file_name = re.findall(r'\d+\w', file)
        torch.save(embeddings, output_path + file_name[0] + '.pt')


main()

print(dict(zip(node_type_dict.keys(), node_index_dict.values())))