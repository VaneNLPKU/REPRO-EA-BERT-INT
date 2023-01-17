import pickle
from transformers import BertTokenizer
import logging
from Param import *
import pickle
import numpy as np
import re
import random
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)



def get_name(string):
    if r"resource/" in string:
        sub_string = string.split(r"resource/")[-1]
    elif r"property/" in string:
        sub_string = string.split(r"property/")[-1]
    else:
        sub_string = string.split(r"/")[-1]
    sub_string = sub_string.replace('_',' ')
    return sub_string



def ent2desTokens_generate(Tokenizer,des_dict_path,ent_list_1,ent_list_2,des_limit = DES_LIMIT_LENGTH - 2):
    """ent_list_1/2: List[str] -> 每个实体的字面名"""
    print("load desription data from... :", des_dict_path)
    ori_des_dict = pickle.load(open(des_dict_path,"rb"))
    ent2desTokens = dict()
    ent_set_1 = set(ent_list_1)
    ent_set_2 = set(ent_list_2)
    for ent, ori_des in ori_des_dict.items():  # TODO 这个循环的效率极低，应当优化！
        if ent not in ent_set_1 and ent not in ent_set_2:
            continue
        dec_string = ori_des
        encode_indexs = Tokenizer.encode(dec_string)[:des_limit]  # 截取描述文本的前若干个字符 # 这里是126 # TODO：只截取前126个文本未免有些离谱（太少了），这里的描述文本少说快1000了，严重怀疑其密度！
        ent2desTokens[ent] = encode_indexs
    print("The num of entity with description:", len(ent2desTokens.keys()))  # 一共有 39251 个实体有描述文件。
    return ent2desTokens  # Dict[str(实体名称), list[int](实体id的token化)]



def ent2Tokens_gene(Tokenizer,ent2desTokens,ent_list,index2entity,
                                ent_name_max_length = DES_LIMIT_LENGTH - 2):
    ent2tokenids = dict()
    for ent_id in ent_list:  # 快
        ent = index2entity[ent_id]
        if ent2desTokens!= None and ent in ent2desTokens:  # 如果使用过了实体描述文件并且确实有描述
            #if entity has description, use entity description
            token_ids = ent2desTokens[ent]
            ent2tokenids[ent_id] = token_ids  # 这个实体就用这些 token_ids
        else:
            #else, use entity name.
            ent_name = get_name(ent)
            token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
            ent2tokenids[ent_id] = token_ids
    return ent2tokenids  # dict[int(实体id), List[int](token_ids)]



def ent2bert_input(ent_ids,Tokenizer,ent2token_ids,des_max_length=DES_LIMIT_LENGTH):
    ent2data = dict()
    pad_id = Tokenizer.pad_token_id

    for ent_id in ent_ids:
        ent2data[ent_id] = [[],[]]
        ent_token_id = ent2token_ids[ent_id]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] = ent_token_ids
        ent2data[ent_id][1] = ent_mask_ids
    return ent2data  # 每一个实体和它要送给bert的输入 dict[int, tuple[(input_ids, masks)]]






def read_data(data_path = DATA_PATH,des_dict_path = DES_DICT_PATH):
    def read_idtuple_file(file_path):
        """
            加载一个id元组文件，返回一个列表。
        """
        print('loading a idtuple file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret
    def read_id2object(file_paths):
        """
            返回一个字典，从id获取对象。
        """

        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...  ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        """
            返回一个列表，读取某个文件，不需要知道读啥，挺快的。
        """
        print('loading a idx_obj file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret

    print("load data from... :", data_path)
    #ent_index(ent_id)2entity / relation_index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])  # dict[int, str] -> 200: 'http://ja.dbpedia.org/resource/諫早市'
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])  # dict[int, str] -> 201: 'http://ja.dbpedia.org/property/製造者'
    entity2index = {e:idx for idx,e in index2entity.items()}  # index2entity的反向映射
    rel2index = {r:idx for idx,r in index2rel.items()}  # index2rel的反向映射

    #triples
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')  # 由id构成的三元组列表，例如 (5202, 273, 9699)
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')  # 同上
    index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')  # List[Tuple(int, str)] -> (0, 'http://ja.dbpedia.org/resource/アリアンツ・リヴィエラ')
    index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')  # 同上，这两个变量和index2entity类似，但区别是这二者只存储了各自图谱的

    #ill 对齐的实体
    train_ill = read_idtuple_file(data_path + 'sup_pairs')  # 对齐实体之训练集（种子实体） List[Tuple[int, int]] # 目前有4500个。
    test_ill = read_idtuple_file(data_path + 'ref_pairs')  # 对齐实体之测试集 List[Tuple[int, int]] # 目前有10500个。
    ent_ill = []  # 所有的对齐实体
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)

    # ent_idx
    entid_1 = [entid for entid,_ in index_with_entity_1]  # list: KG1的所有实体id
    entid_2 = [entid for entid,_ in index_with_entity_2]  # list: KG2的所有实体id
    entids = list(range(len(index2entity)))  # list: 全部实体的id

    # ent2descriptionTokens  #
    Tokenizer = BertTokenizer.from_pretrained("E:\我的科研仓库\预训练模型权重\\bert-base-chinese")
    if des_dict_path!= None:  # 如果存在实体描述文件，则读取
        ent2desTokens = ent2desTokens_generate(Tokenizer,des_dict_path,[index2entity[id] for id in entid_1],[index2entity[id] for id in entid_2])  # return: Dict[str(实体名称), list[int](实体id的token化)]
    else:
        ent2desTokens = None

    # ent2basicBertUnit_input.
    ent2tokenids = ent2Tokens_gene(Tokenizer,ent2desTokens,entids,index2entity)  # dict[int(实体id), List[int](token_ids)]
    ent2data = ent2bert_input(entids,Tokenizer,ent2tokenids)  # dict of 每个 entity_id 构造给bert的输入

    return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2



