import torch
import numpy as np
import re
import torch.nn.functional as F
import torch.nn as nn
import time
from CNN import SimpleCNN1D
import torch.optim as optim
from tqdm.notebook import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
threshold = 0.7

# model = SiameseMLP(100, 128, 64, 32, 16, 1).to(device)
# model = SiameseMLP(100, 64, 32, 16, 1).to(device)
model = SimpleCNN1D(100, 64, 32, 64, 16, 1).to(device)
# model = SiameseLSTM(100, 256, 50).to(device)
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def create_batches(data):
    # random.shuffle(data)
    batches = [data[i:i + 32] for i in range(0, len(data), 32)]
    return batches


def gcj_dataset(id):
    indexdir = '../input/codeclone_gcj/javadata/'
    if id == '0':
        trainfile = open(indexdir + 'trainall.txt')
        validfile = open(indexdir + 'valid.txt')
        testfile = open(indexdir + 'test.txt')
    elif id == '13':
        trainfile = open(indexdir + 'train13.txt')
        validfile = open(indexdir + 'valid.txt')
        testfile = open(indexdir + 'test.txt')
    elif id == '11':
        trainfile = open(indexdir + 'train11.txt')
        validfile = open(indexdir + 'valid.txt')
        testfile = open(indexdir + 'test.txt')
    elif id == '0small':
        trainfile = open(indexdir + 'trainsmall.txt')
        validfile = open(indexdir + 'valid.txt')
        testfile = open(indexdir + 'test.txt')
    elif id == '13small':
        trainfile = open(indexdir + 'train13small.txt')
        validfile = open(indexdir + 'validsmall.txt')
        testfile = open(indexdir + 'testsmall.txt')
    elif id == '11small':
        trainfile = open(indexdir + 'train11small.txt')
        validfile = open(indexdir + 'validsmall.txt')
        testfile = open(indexdir + 'testsmall.txt')
    else:
        print('file not exist')
        quit()
    trainlist = trainfile.readlines()
    validlist = validfile.readlines()
    testlist = testfile.readlines()

    return trainlist, validlist, testlist


def get_dataset(id):
    indexdir = '../input/codeclone_bcb/BCB/'
    if id == '0':
        trainfile = open(indexdir + 'traindata.txt')
        validfile = open(indexdir + 'devdata.txt')
        testfile = open(indexdir + 'testdata.txt')
    elif id == '11':
        trainfile = open(indexdir + 'traindata11.txt')
        validfile = open(indexdir + 'devdata.txt')
        testfile = open(indexdir + 'testdata.txt')
    else:
        print('file not exist')
        quit()
    trainlist = trainfile.readlines()
    print(len(trainlist))
    validlist = validfile.readlines()
    print(len(validlist))
    testlist = testfile.readlines()
    print(len(testlist))
    print("data total number:", len(trainlist) + len(validlist) + len(testlist))

    return trainlist, validlist, testlist


def train_and_valid(trainlist, validlist):
    epochs = trange(3, leave=True, desc="Epoch")
    best_f1 = None
    total_train_time = 0
    total_valid_time = 0
    final_loss = 0
    count = 0
    for epoch in epochs:
        batches = create_batches(trainlist)
        totalloss = 0.0
        main_index = 0.0
        train_time_start = time.time()
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            optimizer.zero_grad()
            batchloss = 0
            for line in batch:
                pairinfo = line.split()
                code1path = '../input/codeclone_bcb/BCB' + pairinfo[0].strip('.')
                code2path = '../input/codeclone_bcb/BCB' + pairinfo[1].strip('.')
                label = int(pairinfo[2])
                embeddings_dir = 'processed_data/embeddings_3/'  # embeddings_2是自监督序列嵌入，embeddings_3是词向量序列嵌入，4是随机游走嵌入

                result_1 = re.findall(r'\d+\w', code1path)
                result_2 = re.findall(r'\d+\w', code2path)

                code1_embedding_path = embeddings_dir + result_1[0] + '.pt'
                code2_embedding_path = embeddings_dir + result_2[0] + '.pt'

                #                 code1path = pairinfo[0].replace('\\', '/')
                #                 code2path = pairinfo[1].replace('\\', '/')
                #                 label=int(pairinfo[2])

                #                 embeddings_dir = 'processed_data/GCJ/embeddings_1/' # embeddings是自监督嵌入，embeddings_1是词向量嵌入，2是随机游走嵌入

                #                 code1_embedding_path = embeddings_dir + code1path + '.pkl.pt'
                #                 code2_embedding_path = embeddings_dir + code2path + '.pkl.pt'

                code1_feature = torch.load(code1_embedding_path).to(device)
                code2_feature = torch.load(code2_embedding_path).to(device)

                #                 print(code1_feature)

                if label == -1:
                    label = 0

                label = torch.tensor(label, dtype=torch.float, device=device)
                label = torch.unsqueeze(label, 0)
                #                 vector = torch.tensor(vector, device=device)

                output = model(code1_feature, code2_feature)
                output = torch.squeeze(output, 0)
                #                 print(output)
                batchloss = batchloss + criterion(output, label)

            #             batchloss.backward(retain_graph=True)
            batchloss.backward()
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
            loss = totalloss / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

        train_time_end = time.time()
        train_time = train_time_end - train_time_start
        total_train_time = total_train_time + train_time

        valid_time_start = time.time()
        devresults, valid_p, valid_r, valid_f1, valid_tp, valid_tn, valid_fp, valid_fn = test(validlist)
        valid_time_end = time.time()
        valid_time = valid_time_end - valid_time_start
        total_valid_time = total_valid_time + valid_time
        devfile = open('result/nn_dev/dev_epoch_' + str(epoch + 1), mode='w')
        for res in devresults:
            devfile.write(str(res) + '\n')
        devfile.close()
        if best_f1 is None or best_f1 < valid_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), 'models/nn/best_model')

        final_loss = totalloss / main_index

        devfile1 = open('result/nn_dev/nn', mode='w')
        devfile1.write(str(valid_tp) + ' ')
        devfile1.write(str(valid_tn) + ' ')
        devfile1.write(str(valid_fp) + ' ')
        devfile1.write(str(valid_fn) + '\n')
        devfile1.write('loss:' + str(final_loss) + '\n')
        devfile1.write(str(valid_p) + '\n')
        devfile1.write(str(valid_r) + '\n')
        devfile1.write(str(valid_f1) + '\n')
        devfile1.close()

    return total_train_time, total_valid_time, final_loss


def test(testlist):
    # model.eval()
    count = 0
    correct = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results = []
    for line in testlist:
        pairinfo = line.split()
        code1path = '../input/codeclone_bcb/BCB' + pairinfo[0].strip('.')
        code2path = '../input/codeclone_bcb/BCB' + pairinfo[1].strip('.')
        label = int(pairinfo[2])
        embeddings_dir = 'processed_data/embeddings_3/'

        result_1 = re.findall(r'\d+\w', code1path)
        result_2 = re.findall(r'\d+\w', code2path)

        code1_embedding_path = embeddings_dir + result_1[0] + '.pt'
        code2_embedding_path = embeddings_dir + result_2[0] + '.pt'
        #         code1path = pairinfo[0].replace('\\', '/')
        #         code2path = pairinfo[1].replace('\\', '/')
        #         label=int(pairinfo[2])

        #         embeddings_dir = 'processed_data/GCJ/embeddings_1/'

        #         code1_embedding_path = embeddings_dir + code1path + '.pkl.pt'
        #         code2_embedding_path = embeddings_dir + code2path + '.pkl.pt'

        code1_feature = torch.load(code1_embedding_path).to(device)
        code2_feature = torch.load(code2_embedding_path).to(device)

        if label == -1:
            label = 0

        label = torch.tensor(label, dtype=torch.float, device=device)

        output = model(code1_feature, code2_feature)
        output = torch.squeeze(output, 0)
        results.append(output.item())
        prediction = output.item()

        if prediction > threshold and label.item() == 1:
            tp += 1
            # print('tp')
        if prediction <= threshold and label.item() == 0:
            tn += 1
            # print('tn')
        if prediction > threshold and label.item() == 0:
            fp += 1
            # print('fp')
        if prediction <= threshold and label.item() == 1:
            fn += 1
            # print('fn')
    print(tp, tn, fp, fn)
    p = 0.0
    r = 0.0
    f1 = 0.0
    if tp + fp == 0:
        print('precision is none')
        return
    p = tp / (tp + fp)
    if tp + fn == 0:
        print('recall is none')
        return
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print('precision')
    print(p)
    print('recall')
    print(r)
    print('F1')
    print(f1)
    return results, p, r, f1, tp, tn, fp, fn


def main():
    trainlist, validlist, testlist = get_dataset('11')  # BCB
    #     trainlist, validlist, testlist = gcj_dataset('11') #GCJ
    train_time, valid_time, loss = train_and_valid(trainlist, validlist)

    test_time_start = time.time()
    model.load_state_dict(torch.load('models/nn/best_model'))
    testresults, test_p, test_r, test_f1, test_tp, test_tn, test_fp, test_fn = test(testlist)
    test_time_end = time.time()
    test_time = test_time_end - test_time_start

    finalfile = open('result/nn', mode='w')
    #     finalfile = open('result/nn_gcj', mode='w')
    finalfile.write('threshold: ' + str(threshold) + '\n')
    finalfile.write('train_time:' + str(train_time) + '\n')
    finalfile.write('valid_time:' + str(valid_time) + '\n')
    finalfile.write('test_time:' + str(test_time) + '\n')
    finalfile.write(str(test_tp) + ' ')
    finalfile.write(str(test_tn) + ' ')
    finalfile.write(str(test_fp) + ' ')
    finalfile.write(str(test_fn) + '\n')
    finalfile.write('loss:' + str(loss) + '\n')
    finalfile.write(str(test_p) + '\n')
    finalfile.write(str(test_r) + '\n')
    finalfile.write(str(test_f1) + '\n')
    finalfile.close()


main()