import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
from models.Loss_bc import HardTripletLoss
from models.networks.UsdlHead import *

class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()
    
    def forward(self, pred, confidence, tgt):
        uncertainty = (1 - confidence)**2
        B = pred.shape[0]#batch数

        return torch.sum(((pred-tgt)**2)*1/(uncertainty**2+1e-6) + uncertainty**2) / B

class VqLayer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25, compute_confidence=False, quantized_output=False):
        super(VqLayer, self).__init__()
        self.embedding_dim = embedding_dim              #码本向量维度
        self.num_embeddings = num_embeddings            #码本大小
        self.commitment_cost = commitment_cost          #承诺代价权重
        self.compute_confidence = compute_confidence    #是否计算置信度
        self.quantized_output = quantized_output        #是否只输出「硬量化结果」, T为False，P为True

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)                #创建可训练的码本
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)#初始化码本权重

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim) #将输入的三维特征 [B, L, D] 展平为二维特征 [B*L, D]；

        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)  
                     - 2 * torch.matmul(flat_x, self.embedding.weight.T)  
                     + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True)) #计算欧氏距离，找每个特征最匹配的码本向量
        
        # 1. 找每个特征帧距离最近的码本向量的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # 2. 初始化全0的one-hot编码矩阵
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        # 3. 将距离最近的码本索引位置设为1，生成one-hot编码
        encodings.scatter_(1, encoding_indices, 1)
        # 4. 矩阵乘法：one-hot编码 × 码本权重 → 得到硬量化后的特征
        quantized = torch.matmul(encodings, self.embedding.weight)
        # 5. 恢复特征的原始形状 [B, L, D]
        quantized = quantized.view_as(x)

        e_latent_loss = F.mse_loss(quantized.detach(), x)#编码器损失
        q_latent_loss = F.mse_loss(quantized, x.detach())#码本损失
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss#总损失

        if not self.quantized_output:#T特征启用,给所有码本向量分配权重，加权求和
            
            similarities = -distances   # 距离越小，相似度越高
            weights = F.log_softmax(similarities, dim=1)    # 相似度归一化为权重
           
            weighted_output = torch.matmul(weights, self.embedding.weight) # 加权求和码本
            weighted_output = weighted_output.view_as(x)# 恢复原始形状


        output_vector = quantized if self.quantized_output else (weighted_output+quantized) / 2

        confidence = None
        if self.compute_confidence:
            if not self.quantized_output:#T-> 软量化模式 → 用【熵】计算置信度
                
                # 计算weights后，还原概率p
                p = torch.exp(weights)  # p = softmax(similarities)，概率分布
                '''原始
                entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1)
                '''
                # 正确计算熵：H = -sum(p * log(p))
                entropy = -torch.sum(p * torch.log(p + 1e-8), dim=1)  # 此处p+1e-8确保为正数
                
                confidence = 1 - entropy / torch.log(torch.tensor(self.num_embeddings, device=x.device))
                confidence = confidence.view(x.size(0), x.size(1))
            else:#P-> 硬量化模式 → 用【距离比】计算置信度

                nearest_distance = distances.gather(1, encoding_indices).squeeze(1)
                max_distance = distances.max(dim=1)[0]
                confidence = 1 - nearest_distance / max_distance
                confidence = confidence.view(x.size(0), x.size(1))
        output_vector = x + (output_vector - x).detach()#梯度截断

        return output_vector, confidence, vq_loss
    
    
    def weighted_inference(self, x, similarity_metric='cosine'):
        flat_x = x.view(-1, self.embedding_dim)
        if similarity_metric == 'cosine':

            norm_x = F.normalize(flat_x, dim=1)
            norm_embeddings = F.normalize(self.embedding.weight, dim=1)
            similarities = torch.matmul(norm_x, norm_embeddings.T)

        elif similarity_metric == 'negative_distance':

            similarities = -((torch.sum(flat_x ** 2, dim=1, keepdim=True)
                              - 2 * torch.matmul(flat_x, self.embedding.weight.T)
                              + torch.sum(self.embedding.weight.T ** 2, dim=0, keepdim=True)))
        else:
            raise ValueError("Unsupported similarity_metric. Choose 'cosine' or 'negative_distance'.")

        weights = F.log_softmax(similarities, dim=1)

        return weights
        
    def calc_loss(self, x, output_vector, vq_loss):
        recon_loss = F.mse_loss(output_vector, x)
        total_loss = recon_loss + vq_loss
        return total_loss
    

class MainModel(nn.Module):
    #原始
    #输入视频特征的维度|隐藏层维度|Transformer多头注意力的头数|Transformer编码器的堆叠层数|解耦后 T/P 特征的维度|向量量化的码本大小
    def __init__(self, input_feature_dim=1024, hidden_dim=256, nhead=4, num_layers=2, decopuling_dim=128, num_quantized_embedding=64):
        super(MainModel, self).__init__()

        # 解耦层：将输入特征解耦为T和P两路特征
        self.decoupling_T = nn.Sequential(#动作时序特征
            nn.Linear(input_feature_dim, decopuling_dim)
        )
        self.decoupling_P = nn.Sequential(#动作质量特征
            nn.Linear(input_feature_dim, decopuling_dim)
        )

        # 向量量化层：对解耦后的T/P特征分别做量化
        self.vector_quantized_T = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=True)
        self.vector_quantized_P = VqLayer(embedding_dim=decopuling_dim, num_embeddings=num_quantized_embedding, compute_confidence=True, quantized_output=False)
        
        # 备用回归器（论文预留/消融实验用，本项目未实际调用）
        self.weight_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.clip_score_regressor = nn.Sequential(
            nn.Linear(decopuling_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Transformer编码器：捕捉时序依赖，融合T/P特征
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=decopuling_dim*2,#输入维度 = 128+128 = 256 → 因为要把 T 和 P 特征拼接在一起
                nhead=nhead,
                dim_feedforward=hidden_dim,
                activation="relu",
                batch_first=True,
                dropout=0.1 # 添加Dropout，抑制过拟合
            ),
            num_layers=num_layers
        )
        
        # 最终预测头：分数回归+置信度回归
        self.score_regressor = nn.Sequential(
            nn.Linear(decopuling_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout，抑制过拟合
            nn.Linear(hidden_dim, 1),          
        )
        self.confidence_regressor = nn.Sequential(
            nn.Linear(decopuling_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.pred_loss = ModelLoss()#预测损失
        self.tri_loss = HardTripletLoss(margin=0.5, hardest=True)#三元组损失

    def forward(self, f, tgt=None):
        # 步骤1：初始化变量 + 解析输入特征形状
        B, L, E = f.shape#特征的batch数、序列长度、维度
        loss = 0.0
        confidence = 0.0
        
        # 步骤2：特征展平 + 核心解耦：分离T(时序)和P(质量)特征
        f = f.reshape(B*L, -1)#把输入的三维特征展平为二维
        T = self.decoupling_T(f).reshape(B, L, -1)
        P = self.decoupling_P(f).reshape(B, L, -1)

        # 步骤3：三元组损失计算：强制约束T/P特征解耦分离
        T0, P0 = T.reshape(B*L, -1), P.reshape(B*L, -1)
       
        # 拼接 T0 和 P0 作为两类样本（总样本数为 2*B*L）
        output_seperation = torch.cat([T0, P0], dim=0)  # 形状：(2*B*L, decopuling_dim)
        # 生成一维标签：前 B*L 个样本为 0 类，后 B*L 个样本为 1 类
        seperation_label = torch.cat([
            torch.zeros(B*L, dtype=torch.long),  # T0 对应的标签
            torch.ones(B*L, dtype=torch.long)    # P0 对应的标签
        ], dim=0).to(f.device)  # 确保与特征在同一设备
        
        

        # 步骤4：向量量化：对解耦后的T/P特征做量化，累加量化损失
        T, confidence_T, loss_T = self.vector_quantized_T(T)
        P, confidence_P, loss_P = self.vector_quantized_P(P)

        
        loss_tri = self.tri_loss(output_seperation, seperation_label)
        loss += loss_tri * 0.5 # TripletLoss权重从1→0.1     
        loss += loss_T * 0.5  # 0.4label
        loss += loss_P * 0.5  # 0.4label

        # 步骤5：Transformer时序编码：融合T/P特征，捕捉时序依赖
        use_transformer = True
        if use_transformer:
            encoding = torch.cat([T, P], dim=-1)

            f_transformer = self.transformer_encoder(encoding)

             # 步骤6：特征聚合 + 预测动作分数 + 融合置信度
           
            f_transformer = f_transformer[:, 0, :]  #取第一个时间步

            pred = self.score_regressor(f_transformer).squeeze(1)

            confidence = self.confidence_regressor(f_transformer).squeeze(1)

            #confidence = torch.mean(confidence_T*confidence_P, dim=-1).squeeze()
            confidence = torch.mean((confidence_T+confidence_P)/2, dim=-1).squeeze()
            # 步骤7：训练阶段累加预测损失（测试阶段无tgt，跳过）
            if tgt is not None:
                loss += self.pred_loss(pred, confidence, tgt)
        # 步骤8：返回最终结果
        return pred, confidence, loss

if __name__ == '__main__':
    device = torch.device('cuda:0')

    model = MainModel().to(device)
    inputs = torch.randn(4, 192, 1024).to(device)
    tgt = torch.randn(4, 1).to(device)
    outputs, confidence, loss = model(inputs, tgt)
    loss.backward()
    print(outputs.shape, confidence.shape, loss)