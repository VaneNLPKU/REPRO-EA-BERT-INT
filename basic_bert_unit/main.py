import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
import torch.optim as optim
import random

from Read_data_func import read_data
from Param import *
from Basic_Bert_Unit_model import Basic_Bert_Unit_model
from Batch_TrainData_Generator import Batch_TrainData_Generator
from train_func import train
import numpy as np


def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    # read data
    print("start load data....")
    ent_ill, train_ill, test_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    ent2data, rel_triples_1, rel_triples_2 = read_data()  # 加载数据
    print("---------------------------------------")

    # 加载模型
    Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM)  # 加载BERT模型
    Model.cuda(CUDA_NUM)  # cuda: 0


    print("all entity ILLs num:", len(ent_ill))
    print("rel num:", len(index2rel))
    print("ent num:", len(index2entity))
    print("triple1 num:", len(rel_triples_1))
    print("triple2 num:", len(rel_triples_2))

    # get train/test_ill
    if RANDOM_DIVIDE_ILL:  # 这里默认是 False
        print(RANDOM_DIVIDE_ILL)  # ill
        # get train/test_ILLs by random divide all entity ILLs
        print("Random divide train/test ILLs!")
        random.shuffle(ent_ill)
        train_ill = random.sample(ent_ill, int(len(ent_ill) * TRAIN_ILL_RATE))
        test_ill = list(set(ent_ill) - set(train_ill))
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL num:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL num:", len(set(train_ill) & set(test_ill)))
    else:
        # get train/test ILLs from file.
        print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL:", len(set(train_ill) & set(test_ill)))

    Criterion = nn.MarginRankingLoss(MARGIN, size_average=True)  # todo 看看这个loss是个啥
    Optimizer = AdamW(Model.parameters(), lr=LEARNING_RATE)
    # ent_ill 是所有的对齐实体
    ent1 = [e1 for e1, e2 in ent_ill]  # 对齐实体中e1的id集合
    ent2 = [e2 for e1, e2 in ent_ill]  # 对齐实体中e2的id集合

    # training data generator(can generate batch-size training data)
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, index2entity, batch_size=TRAIN_BATCH_SIZE,
                                           neg_num=NEG_NUM)

    train(Model, Criterion, Optimizer, Train_gene, train_ill, test_ill, ent2data)  # 开训！


if __name__ == '__main__':
    fixed(SEED_NUM)  # 固定所有的seed，确保程序一致性可控。
    main()
