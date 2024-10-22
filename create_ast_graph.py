import javalang
from javalang.ast import Node
import os
from anytree import AnyNode, RenderTree
import networkx as nx

token_type = ('DecimalInteger', 'HexInteger', 'Identifier', 'OctalInteger', 'String', 'DecimalFloatingPoint',
              'BasicType', 'HexFloatingPoint')

def parse_code(fileStr):
    programfile = open(fileStr, encoding='utf-8')
    #     print(os.path.join(rt,file))
    programtext = programfile.read()
    #     programtext=programtext.replace('\r','')
    programtokens = javalang.tokenizer.tokenize(programtext)
    #     print(list(programtokens))
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()  # 这一步就是ast
#     programast = javalang.parser.parse(programtokens)

    #     print(programast)
    programfile.close()

    # file = open(fileStr, "r", encoding='utf-8')
    # tokens = list(javalang.tokenizer.tokenize(file.read()))
    tokens = list(javalang.tokenizer.tokenize(programtext))
    # print("programtokens", list(tokens))
    # file.close()
    # print(programast)
    tree = programast
    return tree, tokens


def get_token(node):
    token = ''
    # print(isinstance(node, Node))
    # print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    # print(node.__class__.__name__,str(node))
    # print(node.__class__.__name__, node)
    return token


def get_child(root):
    # print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))


def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    # print(id)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)

# token type dictionary
def get_token_dict(tokens):
    typedict = {}
    for token in tokens:
        token_type = str(type(token))[:-2].split(".")[-1]
        token_value = token.value
        if token_value not in typedict:
            typedict[token_value] = token_type
        else:
            if typedict[token_value] != token_type:
                print('!!!!!!!!')

    return typedict


# def get_sequence(node, sequence):
#     token, children = get_token(node), get_child(node)
#     sequence.append(token)
#     # print(len(sequence), token)
#     for child in children:
#         get_sequence(child, sequence)

def get_sequence(node, typedict, sequence):
    token, children = get_token(node), get_child(node)
    # if len(children) == 0:
    #     # print(token)
    #     if token in typedict.keys() and typedict[token] in token_type:
    #         # token = token + " -> " + typedict[token] + " -> " + "leaf_node"
    #         token = typedict[token]
    # print(token)
    # print(len(children))
    sequence.append(token)
    # print(len(sequence), token)
    for child in children:
        get_sequence(child, typedict, sequence)

def create_ast_and_get_allTokens(tree, typedict):  # 生成ast以及找到所有token
    nodelist = []
    alltokens = []
    # tree, tokens = parse_code(fileStr)
    # typedict = get_token_dict(tokens)
    newtree = AnyNode(id=0, token=None, data=None)
    createtree(newtree, tree, nodelist)
    get_sequence(tree, typedict, alltokens)
    return newtree, alltokens


def getnodeandedge_astonly(node, nodeindexlist, src, tgt, typedict):
    token = node.token
    # nodeindexlist.append([vocabdict[token]])
    # if len(node.children) == 0:
    #     if token in typedict.keys() and typedict[token] in token_type:
    #         if typedict[token] == 'BasicType':
    #             token = 'TypeIdt'
    #         else:
    #             token = typedict[token]
    nodeindexlist.append(token)
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        getnodeandedge_astonly(child, nodeindexlist, src, tgt, typedict)


def find_node_and_edge(fileStr):
    x = []
    edgesrc = []
    edgetgt = []
    tree, tokens = parse_code(fileStr)
    typedict = get_token_dict(tokens)
    # print(typedict)
    newtree, alltokens = create_ast_and_get_allTokens(tree, typedict)
    #     print('allnodes: ', len(alltokens))
    # alltokens = list(set(alltokens))
    # vocabsize = len(alltokens)
    # tokenids = range(vocabsize)
    # vocabdict = dict(zip(alltokens, tokenids))
    getnodeandedge_astonly(newtree, x, edgesrc, edgetgt, typedict)  # 得到结点和边
    return x, edgesrc, edgetgt, alltokens


def create_ast_graph_func(fileStr):
    g_edge = []
    x, edgesrc, edgetgt, alltokens = find_node_and_edge(fileStr)
    for i in range(len(edgesrc)):
        g_edge.append((edgesrc[i], edgetgt[i]))

    #     print("g_edge:", g_edge)
    g = nx.Graph()
    g.add_edges_from(g_edge)  # 生成ast图
    return g, alltokens, x, g_edge