import os
import pickle
import random
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

RANDOM_SEED = 42

TRAIN_SIZE_RATIO = 0.8

IMPLEMENTED_DATASET = ["finefs", "rg", "fis-v", "mtl-aqa"]#数据集种类，共4种
SAMPLE_NUM = {# 各数据集/子集的样本总数
    "finefs":{
        "short program": 729,
        "free skating": 438
    },
    "rg":{
        "ribbon": 250,
        "clubs": 250,
        "hoop": 250, 
        "ball": 250
    },
    "fis-v":{
        "all": 500
    },
    "mtl-aqa":{
        "all": 1412
    }
}
FEATURE_SEQ_LENGTH_FILL_ZERO_TO = {# 特征序列的“标准长度”
    "finefs":{
        "short program": 192,
        "free skating": 256
    },
    "rg":{
        "ribbon": 80,
        "clubs": 80,
        "hoop": 80, 
        "ball": 80
    },
    "fis-v":{
        "all": 304
    },
    "mtl-aqa":{
        "all": 8    
    }
}
MAX_SCORE = {# 各子集评分的最大值（归一化用）
    "finefs":{
        #TES
        #"short program": 65.98,
        #"free skating": 129.14
        #PCS
        "short program": 48.47,
        "free skating": 95.84
    },
    "rg":{
        "ribbon": 21.7,
        "clubs": 24.0,
        "hoop": 23.6, 
        "ball": 23.7
    },
    "fis-v":{
        "all": 80.75
    },
    "mtl-aqa":{
        "all": 104.5
    }
}
MIN_SCORE = {# 各子集评分的最小值（归一化用）
    "finefs":{
        #TES
        #"short program": 16.68,
        #"free skating": 30.37
        #PCS
        "short program": 18.74,
        "free skating": 38.75
    },
    "rg":{
        "ribbon": 5.05,
        "clubs": 6.5,
        "hoop": 3.25, 
        "ball": 3.85
    },
    "fis-v":{
        "all": 16.81
    },
    "mtl-aqa":{
        "all": 0.0
    }
}

FINFFS_FS_START_INDEX = 729 # FineFS free skating样本的起始索引

 # 遍历FineFS的json标签文件，汇总成“索引 评分”的txt台账
 #如果要在TES和PCS间切换，需要删除已经生成的labels.txt
def generate_finefs_labels(label_path):
    label_json_list = os.listdir(r"./data/dataset/FineFS/labels")
    with open(label_path, "w") as f: # 写入标签文件（index score）
        for label_json_file in label_json_list:
            if label_json_file[-5:] == ".json":  # 过滤JSON文件
                index = label_json_file[:-5]    # 文件名作为样本索引（去掉.json）
                # 加载JSON中的分数
                performance_dict = json.load(open(os.path.join(r"./data/dataset/FineFS/labels", label_json_file), "r"))
                #score_value = performance_dict["total_element_score"]#TES
                score_value = performance_dict["total_program_component_score(factored)"]#PCS
                f.write("{} {}\n".format(index, score_value)) # 写入"索引 分数"
    f.close()
    
# 从MTL-AQA的pkl标注文件提取评分，生成txt台账；同时重命名特征文件为连续索引
def generate_mtlaqa_labels(label_path):
    info_dict = pickle.load(open(r"./data/dataset/MTL-AQA/final_annotations_dict.pkl", "rb"))
    data_dir = "./data/dataset/MTL-AQA/datas"
    not_clear_key = list(info_dict.keys()) # 原始索引列表
    with open(label_path, "w") as f:
        for index, (x,y) in enumerate(not_clear_key):# 重新分配连续索引（0,1,2...）
            score_value = info_dict[(x,y)]["final_score"]# 提取最终分数
            f.write("{} {}\n".format(index, score_value))# 写入新索引和分数
            
            # 重命名特征文件（将x_y.pkl改为index.pkl，统一索引）
            old_filename = os.path.join(data_dir, f"{x:02d}_{y:02d}.pkl")  
            new_filename = os.path.join(data_dir, f"{index}.pkl")          
            if os.path.exists(old_filename):
                os.rename(old_filename, new_filename)
                print(f"rename: {old_filename} -> {new_filename}")
            else:
                print(f"Not exiss: {old_filename}")
            
    f.close()

# 类初始化
class AqaDataset(Dataset):
    def __init__(self, dataset_used="finefs", subset=None):
        print("Initializing dataset...")
        self.init_dataset(dataset_used)    # 初始化数据集
        self.init_subset(subset)           # 初始化子任务（如FineFS的short program）
        
        print("Dataset Setting Complete.")
        print("Using dataset: {}".format(self.dataset_used))
        print("Using Subset: {}".format(self.subset))
        
    # 按数据集类型，加载/生成标签文件，构建{索引:评分}的快速查询字典    
    def init_dataset(self, dataset_used):
        dataset_used = dataset_used.lower() # 统一小写（避免大小写错误）
        self.dataset_used = dataset_used


        # 根据数据集种类进行处理
        if dataset_used == "finefs":
            # 定义FineFS标签路径
            self.label_path = r"./data/dataset/FineFS/labels.txt"
            if not os.path.exists(self.label_path):# 标签文件不存在则生成
                generate_finefs_labels(self.label_path)
            self.label_dict = {}
            with open(self.label_path, "r") as f:# 加载标签到字典（index: score）
                for line in f:
                    index, score = line.strip().split()
                    self.label_dict[int(index)] = float(score)
            f.close()
            self.features_dir = r"./data/dataset/FineFS/datas"# 特征目录
        
        elif dataset_used == r"mtl-aqa":
            self.label_path = r"./data/dataset/MTL-AQA/labels.txt"
            if not os.path.exists(self.label_path):
                generate_mtlaqa_labels(self.label_path)
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                for line in f:
                    index, score = line.strip().split()
                    self.label_dict[int(index)] = float(score)
            f.close()
            self.features_dir = r"./data/dataset/MTL-AQA/datas"
        
        elif dataset_used == 'gdlt' or dataset_used == "rg":
            self.dataset_used = "rg"
            self.label_path = r"./data/dataset/RG/labels.txt"
            self.label_dict = {}
            with open(self.label_path, "r") as f:
                next(f)
                for line in f:
                    line_content = line.strip().split()
                    index = line_content[0]
                    #更正:数据集中总分已算了扣分值
                    #score = float(line_content[3])

                    score = float(line_content[3]) - float(line_content[4])
                    
                    self.label_dict[index] = score
            self.features_dir = r"./data/dataset/RG/datas" 
        
        ###实验中未用到###
        #elif dataset_used == r"fis-v":
        #    self.label_path = r"./data/dataset/fis-v/labels.txt"
        #    self.label_dict = {}
        #    with open(self.label_path, "r") as f:
        #        for line in f:
        #            index, score1, score2, score3 = line.strip().split()
        #            self.label_dict[int(index)] = float(score1)+float(score2)-float(score3)
        #    f.close()
        #    self.features_dir = r"./data/dataset/fis-v/swintx_avg_fps25_clip32"
        ###
        else:# 默认使用FineFS
            print("Default use dataset: {}".format("FineFS"))
            self.label_path = r"./data/dataset/FineFS"

    def init_subset(self, subset=None):
        if subset is not None:
            subset = subset.lower() # 统一小写
            
        if self.dataset_used == "finefs":
            # 验证子任务是否合法，默认短节目
            if subset not in ["short program", "sp", "free skating", "fs"]:
                print("Subset {} of FineFS not found. Default set as {}.".format(subset, "short program"))
                self.subset = "short program"
            else:
                self.subset = "short program" if subset in ["short program", "sp"] else "free skating"
        elif self.dataset_used == "mtl-aqa":
            self.subset = "all"
        elif self.dataset_used == "fis-v":
            self.subset = "all"
        elif self.dataset_used == "rg":
             # 验证RG子任务，默认ribbon
            if subset not in ["ribbon", "hoop", "clubs", "ball"]:
                print("Subset {} of RG not found. Default set as {}.".format(subset, "ribbon"))
                self.subset = "ribbon"
            else:
                self.subset = subset
        
    
    def __len__(self):
        return SAMPLE_NUM[self.dataset_used][self.subset]
            
    def __getitem__(self, idx):
        feature = None
        label = 0.0
        
        if self.dataset_used == "finefs":
            if self.subset == "short program":
                # 加载短节目特征（pkl文件）
                feature = torch.load(
                    os.path.join(self.features_dir, "{}.pkl".format(idx))
                )
            elif self.subset == "free skating":
                idx += FINFFS_FS_START_INDEX # 自由滑索引偏移
                feature = torch.load(
                    os.path.join(self.features_dir, "{}.pkl".format(idx))
                )
            label = self.label_dict[idx]# 读取标签
            
        elif self.dataset_used == "rg":
            idx += 1
            if self.subset == "ball" and idx >=134:
                idx += 1
            feature_path = os.path.join(self.features_dir, "{}_{:03}.npy".format(self.subset.title(), idx))
            feature = np.load(feature_path)
            feature = torch.from_numpy(feature)
            label = self.label_dict["{}_{:03}".format(self.subset.title(), idx)]
        
        elif self.dataset_used == "fis-v":
            idx += 1
            feature_path = os.path.join(self.features_dir, "{}.npy".format(idx))
            feature = np.load(feature_path)
            feature = torch.from_numpy(feature)
            label = self.label_dict[idx]
        
        elif self.dataset_used == "mtl-aqa":
            feature_path = os.path.join(self.features_dir, "{}.pkl".format(idx))
            feature = pickle.load(open(feature_path, "rb"))

            # ========== 新增这1行 核心修复代码 ==========
            feature = torch.from_numpy(feature).float()  # numpy数组转Tensor + 转float32类型
            # ==========================================
            L, N = feature.shape 
            zero_to_append = torch.zeros(L, 1024-N)
            feature = torch.cat((feature, zero_to_append), dim=1)

            label = self.label_dict[idx]
        
        # 特征序列填充到固定长度（补零）：(0,0)表示维度1/2不补，(0, target_len - 当前长度)表示维度0补零
        feature =  F.pad(feature, (0, 0, 0, FEATURE_SEQ_LENGTH_FILL_ZERO_TO[self.dataset_used][self.subset]-feature.shape[0]), mode="constant", value=0.0)
        # 分数归一化到[0,1]
        min_score = MIN_SCORE[self.dataset_used][self.subset]
        max_score = MAX_SCORE[self.dataset_used][self.subset]
        
        label = (label - min_score) / (max_score - min_score)
       
        return feature, label # 返回（特征张量，归一化标签）

# 主函数
if __name__ == "__main__":

    dataset_example = AqaDataset(dataset_used="finefs",)
    print("Total nums of {}:{} is {}".format(dataset_example.dataset_used, dataset_example.subset, len(dataset_example)))
    max_seq_length = 0
    max_score = 0
    min_score = 100000
    for i in range(len(dataset_example)):
        feature, score = dataset_example[i]
        max_seq_length = max(max_seq_length, feature.shape[0]) # 统计最大序列长度
        max_score = max(max_score, score)# 统计最大归一化分数
        min_score = min(min_score, score)# 统计最小归一化分数
        
    print("Max length is: {}, Max score is: {}, Min score is: {}".format(max_seq_length, max_score, min_score))