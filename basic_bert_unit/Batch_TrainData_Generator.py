import numpy as np
import torch
import torch.nn as nn
from Param import *
import time


class Batch_TrainData_Generator(object):
    def __init__(self,train_ill,ent_ids1,ent_ids2,index2entity,batch_size,neg_num):
        self.ent_ill = train_ill
        self.ent_ids1 = ent_ids1
        self.ent_ids2 = ent_ids2
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.iter_count = 0
        self.index2entity = index2entity
        print("In Batch_TrainData_Generator, train ill num: {}".format(len(self.ent_ill)))  # 4500
        print("In Batch_TrainData_Generator, ent_ids1 num: {}".format(len(self.ent_ids1)))  # 15000
        print("In Batch_TrainData_Generator, ent_ids2 num: {}".format(len(self.ent_ids2)))  # 15000
        # print("In Batch_TrainData_Generator, keys of index2entity num: {}".format(len(self.index2entity)))




    def train_index_gene(self,candidate_dict):
        """
        generate training data (entity_index). 进行负采样等策略。
        """
        train_index = [] #training data
        candid_num = 999999
        for ent in candidate_dict:  # 将 candidate_dict 的值（候选实体id组成的列表）替换为 np.array
            candid_num = min(candid_num,len(candidate_dict[ent]))
            candidate_dict[ent] = np.array(candidate_dict[ent])
        for pe1, pe2 in self.ent_ill:  #
            for _ in range(self.neg_num):  # 负采样的数量，默认是 2 —— 和下方的代码配合，使得正负采样的比例为1比2，每个实体对一个负样。（看来没有采取负采样策略）
                if np.random.rand() <= 0.5:  # 一半的概率，KG1的实体随机采KG2中的实体作为负例
                    #e1
                    ne1 = candidate_dict[pe2][np.random.randint(candid_num)]
                    ne2 = pe2
                else:  # 一半的概率，KG2的实体随机采KG1中的实体作为负例
                    ne1 = pe1
                    ne2 = candidate_dict[pe1][np.random.randint(candid_num)]
                #same check
                if pe1!=ne1 or pe2!=ne2:  # 同时保证不采样到自己和自己的关系
                    train_index.append([pe1,pe2,ne1,ne2])
        np.random.shuffle(train_index)
        self.train_index = train_index  # 负采样之后的训练样本
        self.batch_num = int(np.ceil( len(self.train_index) * 1.0 / self.batch_size ) )  # 默认情况是8990，即batch的数量（总数除以batch大小）。



    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1

            batch_data = self.train_index[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
            # 每次返回的包括数个正负实体的id
            pe1s = [pe1 for pe1,pe2,ne1,ne2 in batch_data]
            pe2s = [pe2 for pe1,pe2,ne1,ne2 in batch_data]
            ne1s = [ne1 for pe1,pe2,ne1,ne2 in batch_data]
            ne2s = [ne2 for pe1,pe2,ne1,ne2 in batch_data]

            return pe1s,pe2s,ne1s,ne2s

        else:
            del self.train_index
            self.iter_count = 0
            raise StopIteration()
