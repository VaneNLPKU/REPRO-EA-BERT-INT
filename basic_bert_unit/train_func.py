import torch
import torch.nn as nn
import torch.nn.functional as F
from Param import *
import numpy as np
import time
import pickle
from eval_function import cos_sim_mat_generate, batch_topk, hit_res


def entlist2emb(Model, entids, entid2data, cuda_num):
    """
    return basic bert unit output embedding of entities
    返回用基本bert单元编码后的实体表示
    （输入的是id组成的列表，因此可以做到一定程度的利用显存）
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids, batch_mask_ids)  # 就是一批（实体名称或描述）的张量进了BERT之后的输出
    del batch_token_ids
    del batch_mask_ids
    return batch_emb


def generate_candidate_dict(Model, train_ent1s, train_ent2s, for_candidate_ent1s, for_candidate_ent2s,
                            entid2data, index2entity,
                            nearest_sample_num=NEAREST_SAMPLE_NUM, batch_size=CANDIDATE_GENERATOR_BATCH_SIZE):
    """
    返回一个字典，
        键为：每个实体的id
        值为：list: 每个实体的 (128个) 候选的id
    """

    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    with torch.no_grad():
        # langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        for i in range(0, len(train_ent1s), batch_size):  # 【慢】，用于将 KG1 的所有实体转为对应的嵌入向量
            temp_emb = entlist2emb(Model, train_ent1s[i:i + batch_size], entid2data, CUDA_NUM).cpu().tolist()
            train_emb1.extend(temp_emb)
        for i in range(0, len(for_candidate_ent2s), batch_size):
            temp_emb = entlist2emb(Model, for_candidate_ent2s[i:i + batch_size], entid2data, CUDA_NUM).cpu().tolist()
            for_candidate_emb1.extend(temp_emb)

        # language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        for i in range(0, len(train_ent2s), batch_size):  # 【慢】，同上。
            temp_emb = entlist2emb(Model, train_ent2s[i:i + batch_size], entid2data, CUDA_NUM).cpu().tolist()
            train_emb2.extend(temp_emb)
        for i in range(0, len(for_candidate_ent1s), batch_size):
            temp_emb = entlist2emb(Model, for_candidate_ent1s[i:i + batch_size], entid2data, CUDA_NUM).cpu().tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()

        # cos sim  # 截止本步骤，已经占用了 1405 /  6144 MB 的GPU
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1, for_candidate_emb1)  # 生成(4500， 15000)相似度大矩阵
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2, for_candidate_emb2)  # 生成(4500， 15000)相似度大矩阵
        torch.cuda.empty_cache()
        # topk index
        _, topk_index_1 = batch_topk(cos_sim_mat1, topn=nearest_sample_num, largest=True)
        topk_index_1 = topk_index_1.tolist()  # 寻找最接近的candidate作为替代（每个实体找了128个候选实体）
        _, topk_index_2 = batch_topk(cos_sim_mat2, topn=nearest_sample_num, largest=True)
        topk_index_2 = topk_index_2.tolist()  # 寻找最接近的candidate作为替代（每个实体找了128个候选实体）
        # get candidate 为每个实体生成候选，这里每个实体找了128个候选。
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)
        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)

        # show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    print("get candidate using time: {:.3f}".format(time.time() - start_time))  # 获取所有的实体及其候选之后的总耗时
    torch.cuda.empty_cache()
    return candidate_dict


def train(Model, Criterion, Optimizer, Train_gene, train_ill, test_ill, entid2data):
    print("start training...")
    for epoch in range(EPOCH_NUM):
        print("+++++++++++")
        print("Epoch: ", epoch)
        print("+++++++++++")
        # generate candidate_dict
        # (candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1, e2 in train_ill]
        train_ent2s = [e2 for e1, e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1  # 对齐实体中e1的id集合
        for_candidate_ent2s = Train_gene.ent_ids2  # 对齐实体中e2的id集合
        print("train ent1s num: {} train ent2s num: {} for_Candidate_ent1s num: {} for_candidate_ent2s num: {}"
              .format(len(train_ent1s), len(train_ent2s), len(for_candidate_ent1s), len(for_candidate_ent2s)))  # train ent1s num: 4500 train ent2s num: 4500 for_Candidate_ent1s num: 15000 for_candidate_ent2s num: 15000
        candidate_dict = generate_candidate_dict(Model, train_ent1s, train_ent2s, for_candidate_ent1s,
                                                 for_candidate_ent2s, entid2data, Train_gene.index2entity)  # 获取每个实体的候选id
        Train_gene.train_index_gene(candidate_dict)  # generate training data with candidate_dict

        # train
        epoch_loss, epoch_train_time = ent_align_train(Model, Criterion, Optimizer, Train_gene, entid2data)
        Optimizer.zero_grad()
        torch.cuda.empty_cache()  # 这个操作可以有效地降低显存占用
        print("Epoch {}: loss {:.3f}, using time {:.3f}".format(epoch, epoch_loss, epoch_train_time))
        if epoch >= 0:
            if epoch != 0:
                save(Model, train_ill, test_ill, entid2data, epoch)
            # test(Model,train_ill,entid2data,TEST_BATCH_SIZE,context="EVAL IN TRAIN SET")
            test(Model, test_ill, entid2data, TEST_BATCH_SIZE, context="EVAL IN TEST SET:")


def test(Model, ent_ill, entid2data, batch_size, context=""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1, e2 in ent_ill]
        ents_2 = [e2 for e1, e2 in ent_ill]

        emb1 = []
        for i in range(0, len(ents_1), batch_size):
            batch_ents_1 = ents_1[i: i + batch_size]
            batch_emb_1 = entlist2emb(Model, batch_ents_1, entid2data, CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0, len(ents_2), batch_size):
            batch_ents_2 = ents_2[i: i + batch_size]
            batch_emb_2 = entlist2emb(Model, batch_ents_2, entid2data, CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1, emb2, batch_size, cuda_num=CUDA_NUM)
        score, top_index = batch_topk(res_mat, batch_size, topn=TOPK, largest=True, cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time() - start_time))
    print("--------------------")


def save(Model, train_ill, test_ill, entid2data, epoch_num):
    print("Model {} save in: ".format(epoch_num),
          MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + "model_epoch_" + str(epoch_num) + '.p')
    Model.eval()
    torch.save(Model.state_dict(), MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + "model_epoch_" + str(epoch_num) + '.p')
    other_data = [train_ill, test_ill, entid2data]
    pickle.dump(other_data, open(MODEL_SAVE_PATH + MODEL_SAVE_PREFIX + 'other_data.pkl', "wb"))
    print("Model {} save end.".format(epoch_num))


def ent_align_train(Model, Criterion, Optimizer, Train_gene, entid2data):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s, pe2s, ne1s, ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model, pe1s, entid2data, cuda_num=CUDA_NUM)  # (1, 300) 的 tensor，其实就是该实体的 embedding。
        pos_emb2 = entlist2emb(Model, pe2s, entid2data, cuda_num=CUDA_NUM)  # 这里一个操作就占用了100M的显存
        batch_length = pos_emb1.shape[0]  # 批大小就1哈
        pos_score = F.pairwise_distance(pos_emb1, pos_emb2, p=1, keepdim=True)  # 小，L1 distance
        del pos_emb1  # 这货删除之后没有释放显存
        del pos_emb2

        neg_emb1 = entlist2emb(Model, ne1s, entid2data, cuda_num=CUDA_NUM)
        neg_emb2 = entlist2emb(Model, ne2s, entid2data, cuda_num=CUDA_NUM)
        neg_score = F.pairwise_distance(neg_emb1, neg_emb2, p=1, keepdim=True)
        del neg_emb1
        del neg_emb2

        label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM)  # pos_score < neg_score
        batch_loss = Criterion(pos_score, neg_score, label_y)
        del pos_score
        del neg_score
        del label_y
        batch_loss.backward()  # 这一步导致显存骤增400M左右，总共占用数量级为 G 。
        Optimizer.step()

        all_loss += batch_loss.item() * batch_length
        all_using_time = time.time() - start_time
    return all_loss, all_using_time
