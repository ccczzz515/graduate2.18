import sys
sys.path.append(".")
import argparse
from tensorboardX import SummaryWriter
import time
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 新增：导入余弦退火调度器
from tqdm import tqdm
import logging
import argparse
import random
from data.aqa_dataset import AqaDataset
from models.networks.main_model import MainModel
from experiments.tools.train import freeze_params, unfreeze_params
import json

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--config_path",
    help="Config File Path.",
    type=str,  
    default="experiments/config/test_vq_model.json"
)
parser.add_argument(
    "--reset_log",
    help="Delete old log and tensorboard.",
    action="store_true",  
    default=False
)
args = parser.parse_args()

print("Loading Config From {} And Writing into Log File...".format(args.config_path))

# 加载配置文件
config = json.load(open(args.config_path, "r"))  
exp_name = config["exp_name"]
log_path = os.path.join(r"experiments/log", "{}.log".format(exp_name))
tensorboard_path = "experiments/log/tensorboard_{}".format(exp_name)
if args.reset_log:
    os.system("rm {} -f".format(log_path))
    os.system("rm {} -rf".format(tensorboard_path))
# 日志初始化
logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
# TensorBoard初始化
tensorboard_writer = SummaryWriter(log_dir = tensorboard_path)
logging.info("New Task Started...")
logging.info("Experiment config:")
logging.shutdown()
os.system("cat {}>>{}".format(args.config_path, log_path))
os.system("echo  >> {}".format(log_path))


from experiments.tools.random_seed import setup_seed
# 固定随机种子
setup_seed(config["random_seed"])

cuda_idx = config["gpu_idx"]
# 选择设备
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
# 模型实例化
model = MainModel(decopuling_dim=config["decopuling_dim"]).to(device)

# 定义3组优化器（适配不同训练阶段）
optimizer_pretrain_all = optim.AdamW(
    model.parameters(),
    lr=config["pretrain_lr"],weight_decay=1e-4)
optimizer_pretrain_unlabel = optim.AdamW(
    list(model.decoupling_P.parameters()) + \
    list(model.decoupling_T.parameters()) + \
    list(model.vector_quantized_P.parameters()) + \
    list(model.vector_quantized_T.parameters()), 
    lr=config["pretrain_lr"]*0.1,weight_decay=1e-4)

# 新增：预训练阶段的余弦退火调度器
scheduler_pretrain_all = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_pretrain_all,
    T_max=config["pretrain_epochs"],  # 余弦周期=预训练总轮数（200）
    eta_min=1e-7  # 预训练最小学习率，避免降到0
)
scheduler_pretrain_unlabel = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_pretrain_unlabel,
    T_max=config["pretrain_epochs"],
    eta_min=1e-8
)

optimizer_train = optim.AdamW(
    [
        {"params":list(model.weight_regressor.parameters())+list(model.clip_score_regressor.parameters())+list(model.transformer_encoder.parameters())+list(model.score_regressor.parameters())+list(model.confidence_regressor.parameters())},
        {"params": list(model.decoupling_P.parameters()) + list(model.decoupling_T.parameters()) + list(model.vector_quantized_P.parameters()) + list(model.vector_quantized_T.parameters()), "lr":config["lr"]},
    ],
    lr = config["pretrain_lr"],weight_decay=1e-4
)
# 新增：主训练阶段的余弦退火调度器
scheduler_train = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_train,
    T_max=config["epochs"],  # 余弦周期=主训练总轮数（2000）
    eta_min=1e-8  # 主训练最小学习率，适配你的lr=5e-6
)
dataset_train_ratio = config["train_ratio"]
train_labeled_ratio = config["labeled_sample_ratio"]
main_dataset = config["main_dataset"]
sub_dataset = config["sub_dataset"]
B = config["batch_size"]
# 加载数据集
dataset = AqaDataset(dataset_used=main_dataset, subset=sub_dataset)

total_sample_num = len(dataset)
train_labeled_sample_num = int(total_sample_num*dataset_train_ratio*train_labeled_ratio)
train_unlabeled_sample_num = int(total_sample_num*dataset_train_ratio) - train_labeled_sample_num
test_sample_num = total_sample_num - train_labeled_sample_num - train_unlabeled_sample_num
# 拆分数据集：带标签训练集、无标签训练集、测试集
train_labeled_dataset, train_unlabeled_dataset, test_dataset = random_split(dataset, lengths=[train_labeled_sample_num, train_unlabeled_sample_num, test_sample_num])
logging.info("Nums of samples: Labeled Training: {}, Unlabled Training: {}, Test: {}".format(
    len(train_labeled_dataset), len(train_unlabeled_dataset), len(test_dataset)))


# 构建DataLoader（批次加载）
train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=B)
train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=B)
test_loader = DataLoader(test_dataset, batch_size=B)

# 预训练函数
def pretrain_one_step():
    model.train()
    loss_val = []
    
    # 训练带标签数据（优化所有参数）
    for feature, _ in train_labeled_loader:
        
        feature = feature.to(device)
        optimizer_pretrain_all.zero_grad()# 梯度清零
        pred, confidence, loss = model(feature)# 前向传播
        loss.backward()# 反向传播（算参数调整方向）
        optimizer_pretrain_all.step()# 更新参数

        loss_val.append(loss.item())
        
        
    # 训练无标签数据（优化部分参数）
    for feature, _ in train_unlabeled_loader:
        
        feature = feature.to(device)
        optimizer_pretrain_unlabel.zero_grad()#同上
        pred, confidence, loss = model(feature)
        loss.backward()
        optimizer_pretrain_unlabel.step()

        loss_val.append(loss.item())
        

    
    # 每10轮记录损失
    if (epoch+1) % 10 == 0:
        loss = sum(loss_val)/len(loss_val)
        logging.info(f"Loss Value:{loss:.4f}")
    

# 有监督训练函数
def supervised_train_one_step():
    model.train()
    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        optimizer_train.zero_grad()
        pred, confidence, loss = model(feature, tgt)# 传入真实标签tgt
        loss.backward()
        optimizer_train.step()
        
# 半监督训练函数
def semi_supervised_train_one_step(current_threshold):
    model.train()
   
    # 先做有监督训练（同上）
    for feature, tgt in train_labeled_loader:
        feature = feature.to(device)
        tgt = tgt.to(device)
        
        optimizer_train.zero_grad()
        pred, confidence, loss = model(feature, tgt)
        loss.backward()
        optimizer_train.step()

    # 再处理无标签数据
    for feature, _ in train_unlabeled_loader:
        feature = feature.to(device)
        
        optimizer_train.zero_grad()# 梯度清零
        with torch.no_grad():# 无梯度（只预测、不调整）
            pred, confidence, _ = model(feature)
        ###
        # 打印confidence的均值、最大值、最小值
        print(f"Confidence mean: {confidence.mean().item():.4f}, "f"max: {confidence.max().item():.4f}, "f"min: {confidence.min().item():.4f}") 
        ###

        # 筛选高置信度样本
        high_confidence_mask = confidence > current_threshold
        high_confidence_features = feature[high_confidence_mask]
        high_confidence_preds = pred[high_confidence_mask]
        
       
        if len(high_confidence_features) > 0:
            
            # 用伪标签训练
            pseudo_labels = high_confidence_preds.detach()  
            pred, confidence, pseudo_loss = model(high_confidence_features, pseudo_labels)
            ##
            pseudo_loss *= 0.5  # 降低权重
            ##
            pseudo_loss.backward()
            optimizer_train.step()    

# 评估函数   
def evaluate(dataloader):
    model.eval()  
    true_scores = []  
    predicted_scores = []  
    loss_val = []

    with torch.no_grad():  
        for feature, tgt in dataloader:
            feature = feature.to(device)
            tgt = tgt.to(device)

            pred, confidence, loss = model(feature, tgt)

            true_scores.extend(tgt.cpu().numpy())  
            predicted_scores.extend(pred.cpu().numpy())  
            loss_val.append(loss.item()) 
    try:
        spearman_corr, _ = spearmanr(true_scores, predicted_scores)
    except ValueError:
        spearman_corr = 0.0  # 或其他默认值
    
    return spearman_corr, sum(loss_val[:-1])/(len(loss_val)-1) if len(loss_val)>1 else loss_val[0] # 平均损失
    
#################################训练流程#################################
# 预训练循环
pretrain_epochs = config["pretrain_epochs"]
for epoch in range(pretrain_epochs):
    if (epoch+1) % 100 == 0:
        logging.info(f"Epoch {epoch+1}/{pretrain_epochs}")
    pretrain_one_step()
    # 新增：预训练调度器更新
    scheduler_pretrain_all.step()
    scheduler_pretrain_unlabel.step()

# 主要训练循环（含阈值衰减）
initial_threshold = config["initial_threshold"] 
threshold_decay = config["threshold_decay"] 
min_threshold = config["min_threshold"]
epochs = config["epochs"]
current_threshold = initial_threshold
for epoch in range(epochs):
    
    if (epoch+1) % 10 == 0:
        logging.info(f"Epoch {epoch+1}/{epochs}, Threshold: {current_threshold:.2f}")  

    '''原始版本使用有监督训练函数
    #supervised_train_one_step()
    '''
    semi_supervised_train_one_step(current_threshold)
     # 新增：主训练调度器更新（核心）
    scheduler_train.step()

    # 每10轮评估并记录指标
    if (epoch+1) % 50 == 0:    
        spearman_corr_train_labeled, loss_train_labeled  = evaluate(train_labeled_loader)
        logging.info(f"Trainset Labeled Spearman Correlation: {spearman_corr_train_labeled:.4f}")
        
        spearman_corr_train_unlabeled, loss_train_unlabeled  = evaluate(train_unlabeled_loader)    
        logging.info(f"Trainset Unlabeled Spearman Correlation: {spearman_corr_train_unlabeled:.4f}")
        
        spearman_corr_test, loss_test = evaluate(test_loader)
        logging.info(f"Testset Spearman Correlation: {spearman_corr_test:.4f}")
        
        tensorboard_writer.add_scalar("Labeled_Train_Set_Sp-corr", spearman_corr_train_labeled, epoch+1)
        tensorboard_writer.add_scalar("Unlabeled_Train_Set_Sp-corr", spearman_corr_train_unlabeled, epoch+1)
        tensorboard_writer.add_scalar("Test_Set_Sp-corr", spearman_corr_test, epoch+1)
        tensorboard_writer.add_scalar("Labeled_Train_Set_Loss", loss_train_labeled, epoch+1)
        tensorboard_writer.add_scalar("Unlabeled_Train_Set_Loss", loss_train_unlabeled, epoch+1)
        tensorboard_writer.add_scalar("Test_Set_Loss", loss_test, epoch+1)
    # 阈值线性衰减
    #current_threshold = initial_threshold - (initial_threshold-min_threshold)*epoch/epochs
    #梯度衰减
    current_threshold = max(min_threshold, current_threshold * threshold_decay)