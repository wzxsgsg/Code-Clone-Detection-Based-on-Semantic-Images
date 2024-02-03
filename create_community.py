from community import community_louvain
import networkx as nx
import re
import os
from create_ast_graph import create_ast_graph_func
import json
import pickle
import numpy as np
from tqdm.notebook import tqdm


def louvain_detect(g, x):
    partition = community_louvain.best_partition(g)  # 社区检测
    pvq = list(set(partition.values()))
    #     print("pvq:", pvq)
    #     print("x_length:", len(x))
    val_dict = {}  # 每个分区所对应的token
    val_dict_num = {}  # 每个分区所对应token的序号
    for key in partition.keys():  # 求每个区间所含有的token
        for i in pvq:
            if partition[key] == i:
                if i in val_dict.keys():
                    val_dict[i].append(x[key])
                    val_dict_num[i].append(key)
                else:
                    val_dict[i] = [x[key]]
                    val_dict_num[i] = [key]

    return partition, val_dict, val_dict_num


def get_subgraph_centrality(partition, g):
    G_new = nx.Graph()
    for u, v in g.edges():
        u_part = partition[u]
        v_part = partition[v]
        if u_part != v_part and not G_new.has_edge(u_part, v_part):
            G_new.add_edge(u_part, v_part)

    subgraph_degree_centrality = list(nx.degree_centrality(G_new).values())
    subgraph_betweenness_centrality = list(nx.betweenness_centrality(G_new).values())
    subgraph_closeness_centrality = list(nx.closeness_centrality(G_new).values())
    subgraph_harmonic_centrality = list(nx.harmonic_centrality(G_new).values())
    subgraph_centrality_result = ((np.array(subgraph_degree_centrality) + np.array(
        subgraph_betweenness_centrality) + np.array(subgraph_closeness_centrality) + np.array(
        subgraph_harmonic_centrality)) / 4).tolist()

    subgraph_result = dict(zip(G_new.nodes(), subgraph_centrality_result))
    return subgraph_result


# 得到每个分区的子图，以及子图的结点和边
def get_subgraph(partition, g, x):
    subgraphs = {}
    for node, comm in partition.items():
        if comm not in subgraphs:
            subgraphs[comm] = nx.Graph()
        subgraphs[comm].add_node(node)

    for u, v in g.edges():
        if partition[u] == partition[v]:
            subgraphs[partition[u]].add_edge(u, v)

    # for i, subgraph in enumerate(subgraphs.values()):
    #     nodes_name = []
    #     print(f"Subgraph {i+1} degree centrality: {sum(dict(nx.degree(subgraph)).values())}")
    #     # print(nx.degree_centrality(subgraph))
    #     print("Nodes:", list(subgraph.nodes()))
    #     for j in list(subgraph.nodes()):
    #         nodes_name.append(x[j])
    #     print("Nodes_Name", nodes_name)
    #     print("Edges:", list(subgraph.edges()))
    #     print()

    return subgraphs


def get_nodes_dict(subgraphs, subgraph_result, x):
    nodes_dict = {}
    for i, subgraph_dict in enumerate(subgraphs.items()):
        subgraph_key = subgraph_dict[0]
        subgraph = subgraph_dict[1]
        nodes_name = []
        for j in list(subgraph.nodes()):
            nodes_name.append(x[j])
        nodes_dict[i] = [subgraph, nodes_name, subgraph_result[subgraph_key]]

    return nodes_dict


def create_json_file(filePath, outputPath):
    g, alltokens, x, g_edge = create_ast_graph_func(filePath)  # 得到图、token、结点列表、边列表
    partition, val_dict, val_dict_num = louvain_detect(g, x)
    subgraph_result = get_subgraph_centrality(partition, g)
    subgraphs = get_subgraph(partition, g, x)
    nodes_dict = get_nodes_dict(subgraphs, subgraph_result, x)

    with open(outputPath, 'wb') as file:
        #         json.dump(nodes_dict, file)
        pickle.dump(nodes_dict, file)


def main():
    folder_path = "../input/codeclone_bcb/BCB/bigclonebenchdata"
    outputPath = "processed_data/node_pkl_subgraph/"
    files = os.listdir(folder_path)
    print(len(files))
    error_file = []
    count = 0
    print("start：")
    for index, file in tqdm(enumerate(files), total=len(files), desc="Processing"):
        file_name = re.findall(r'\d+\w', file)
        #         print(file_name)
        try:
            create_json_file(folder_path + '/' + file, outputPath + file_name[0] + '.pkl')
        except:
            error_file.append(file)
            continue

        count += 1
        if count % 10000 == 0:
            print("process：", count)

    print(error_file)


main()