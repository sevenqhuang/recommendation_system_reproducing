import os
import math
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from collections import defaultdict

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# Embedding Layer, 直接使用PyTorch的Embedding实现：torch.nn.Embedding

# self-attention，直接使用PyTorch的MultiheadAttention实现：torch.nn.MultiheadAttention

# FFN，PointWisteFeedForward module
class PointWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.2, dropout_extra=False):
        super().__init__()
        # 论文里面两个W矩阵都是d*d
        self.fc1 = torch.nn.Linear(embed_dim, embed_dim)   
        self.fc2 = torch.nn.Linear(embed_dim, embed_dim)
        # 偏置项，都是d维向量
        self.b1 = torch.nn.Parameter(torch.zeros(embed_dim))
        self.b2 = torch.nn.Parameter(torch.zeros(embed_dim))
        # 论文公式里只有第一层用了激活函数
        self.activation = torch.nn.ReLU()

        self.dropout_extra = dropout_extra
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        # x: [B, L, d]， L是序列长度，d是隐藏层维度
        if self.dropout_extra:
            out = self.fc1(x) + self.b1  # [batch_size, seq_len, embed_dim]
            out = self.activation(out)    # [batch_size, seq_len, embed_dim]
            out = self.dropout1(out)      # [batch_size, seq_len, embed_dim]
            out = self.fc2(out) + self.b2 # [batch_size, seq_len, embed_dim]
            out = self.dropout2(out)      # [batch_size, seq_len, embed_dim]
        else:
            out = self.fc1(x) + self.b1  # [batch_size, seq_len, embed_dim]
            out = self.activation(out)    # [batch_size, seq_len, embed_dim]
            out = self.fc2(out) + self.b2 # [batch_size, seq_len, embed_dim]
        return out

# LayerNorm module，直接使用PyTorch的LayerNorm实现：torch.nn.LayerNorm
    
# SASRec，运行前请设定好超参数
class SASRec(nn.Module):
    def __init__(self, user_num, item_num):
        super().__init__()
        self.user_num = user_num    
        self.item_num = item_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.norm_type = "pre"  # "pre" or "post"，使用pre-norm还是post-norm
        self.maxlen = 200      # 序列最大长度，即论文中的n，对于MovieLens-1M，最大序列长度n被设置为200
        self.embed_dim = 50  # 嵌入维度，即论文中的d（50） 
        self.num_blocks = 4 # 注意力模块的堆叠层数，即论文中为2
        self.num_heads = 2  # 多头注意力机制的头数，论文中只是用了1个头的attention
        self.learning_rate = 0.001 # 学习率论文中设置为0.001
        self.l2_emb = 0.0    # L2正则化系数 
        self.batch_size = 128 # 批大小为128
        self.num_epochs = 200
        self.state_dict_path = ""

        self.dropout_rate = 0.2 # 由于MovieLens-1M数据集的稀疏性，关闭神经元的丢失率为0.2
        self.dropout_extra = False # 有些复现SASRec的代码有额外的dropout层，这里作为可选项

        # +1是因为需要额外的索引0来表示padding位置
        self.item_emb = torch.nn.Embedding(self.item_num+1, self.embed_dim, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.maxlen+1, self.embed_dim, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.sigmoid = torch.nn.Sigmoid()

        self.attention_layers = torch.nn.ModuleList()
        self.attention_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()

        # 因为想对比pre-norm和post-norm的效果，所以不直接做好层的顺序，只累计层的数量
        for _ in range(self.num_blocks):
            # batch_first=True  # 关键参数, 让输入保持 [batch, seq, hidden]
            self.attention_layers.append(
                torch.nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout_rate, batch_first=True)
            )

            self.forward_layers.append(
                PointWiseFeedForward(self.embed_dim, self.dropout_rate, self.dropout_extra)
            )

            self.attention_layernorms.append(
                torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            )

            self.forward_layernorms.append(
                torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            )
    
    # 根据self.norm_type选择前归一化还是后归一化
    def log2feats(self, log_seqs):
        # log_seqs为输入序列，形状： [batch_size, seq_len]
        # 将序列转成embedding表示
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device)) # 输出形状: [batch_size, seq_len, embed_dim]

        # 缩放嵌入向量，乘以嵌入维度的平方根，论文没提，但这是Transformer中的常见做法
        seqs *= self.item_emb.embedding_dim ** 0.5  # 目的是在注意力计算前保持数值稳定性

        # 3. 创建位置编码
        # np.tile用于将数组沿指定的维度重复，从而生成一个新的数组。它非常适合在需要扩展或重复数据时使用
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])  # 形状: [batch_size, seq_len]
        # 生成位置索引矩阵，从1开始到序列长度
        # 例如: 如果seq_len=4, batch_size=2（即log_seqs.shape[0]=2） → [[1,2,3,4], [1,2,3,4]]

        # 处理padding位置
        poss *= (log_seqs != 0)
        # 将padding位置（值为0）的位置编码设为0
        # 避免对padding位置进行位置编码

        # 添加位置编码
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.device))
        if self.dropout_extra:
            seqs = self.emb_dropout(seqs)  # 对嵌入向量应用dropout进行正则化
        
        # 创建注意力掩码
        tl = seqs.shape[1]  # 序列长度

        # 创建上三角掩码矩阵，用于实现因果注意力
        # 确保每个位置只能关注到它之前的位置
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))
        
        for i in range(len(self.attention_layers)):
            # seqs = torch.transpose(seqs, 0, 1) # 早期的 nn.MultiheadAttention 和 nn.Transformer 默认期望输入形状为 [seq_len, batch_size, embed_dim]，新版设置batch_first=True后不需要转置
            if self.norm_type == "pre":
                # Pre-Norm
                seqs_norm = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](seqs_norm, seqs_norm, seqs_norm, attn_mask=attention_mask)
                seqs_ = seqs + mha_outputs
                seqs_next = seqs_ + self.forward_layers[i](self.forward_layernorms[i](seqs_))
            else:  
                # Post-Norm
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs_ = self.attention_layernorms[i](seqs + mha_outputs)
                seqs_next = self.forward_layernorms[i](seqs_ + self.forward_layers[i](seqs_))
        
        return seqs_next  # 输出形状: [batch_size, seq_len, embed_dim]

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        # user_ids: [batch_size, 1]
        # log_seqs: [batch_size, seq_len]，用户历史行为序列 
        # pos_seqs: [batch_size, seq_len]，正样本序列（行为序列的下一个item）
        # neg_seqs: [batch_size, seq_len]，负样本序列

        # 将历史序列转换为特征表示 
        log_feats = self.log2feats(log_seqs) # [batch_size, seq_len, embed_dim]

        # 获取正负样本的嵌入表示
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))  # [batch_size, seq_len, embed_dim]
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))  # [batch_size, seq_len, embed_dim]

        # 计算匹配分数（内积相似度）
        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # [batch_size, seq_len]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # [batch_size, seq_len]

        # """使用用户ID的改进版本"""
        # log_feats = self.log2feats(log_seqs)
        
        # # 添加用户特定的偏置或变换
        # user_embs = self.user_emb(user_ids)  # [batch_size, embed_dim]
        # user_aware_feats = log_feats + user_embs.unsqueeze(1)  # 广播到序列长度
        
        # pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))
        
        # pos_logits = (user_aware_feats * pos_embs).sum(dim=-1)
        # neg_logits = (user_aware_feats * neg_embs).sum(dim=-1)
        
        # 论文里面计算BCE时使用了Sigmoid
        return self.sigmoid(pos_logits), self.sigmoid(neg_logits)

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        # [u]: [1, 1], [seq]: [1, seqlen], item_list: [101]
        log_feats = self.log2feats(log_seqs)  # [1, seqlen, embed_dim]

        # 只取最后一个时间步的特征作为用户当前兴趣表示，预测下一个特征，即要推荐的item的特征表示
        final_feat = log_feats[:, -1, :]  # [1, embed_dim]

        # 改进方案: 使用最后N个位置
        # final_feat = log_feats[:, -3:, :].mean(dim=1)  # 最后3个位置平均

        # 获取候选物品的嵌入向量
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device))  # [101, embed_dim]

        # 计算要推荐的item的特征表示与所有候选物品的匹配分数（内积相似度）
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # [101, 1]

        return self.sigmoid(logits)  # 压缩到0-1之间的分数
    
    def evaluate(self, user_train, user_test, user_num, item_num, inference_only):
        # 论文使用负采样评估Hit Rate@10和NDCG@10
        hit_10 = 0
        ndcg_10 = 0
        total_users = 0

        users = list(user_train.keys())
        if not inference_only:
            # 训练的时候就不评估全部用户了，不然耗时太久
            users = random.sample(users, min(10000, len(users)))
        for u in users:
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(user_train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: 
                    break

            cur_user_items = set(user_train[u])  # 记录当前用户交互过的item，包括0
            cur_user_items.add(0)
            # 负样本从其他用户交互过的item里采样（且不能出现目标用户交互过的item），采样100个
            # 因为数据集里item是从1连续，所以可以用这种方法找到某个item，而不需要另外存储所有用户的item的集合，再从里面找
            item_list = [user_test[u][0]]  # 正样本
            for _ in range(100):  # 负采样100个item
                neg_item = np.random.randint(1, item_num + 1)
                while neg_item in cur_user_items:
                    neg_item = np.random.randint(1, item_num + 1)
                item_list.append(neg_item)

            # 上面代码构造可视化
            # user_train[u] = [1, 2, 3, 4], user_test[u] = [5] maxlen=6
            # seq = [0 0 1 2 3 4]
            # item_list = [5 11 33 896 7 8 ...], len(item_list)=101

            # 预测候选序列的排序，用[]外包起来，当成batch_size=1放入模型进行预测
            # 模型返回预测分数
            predictions = -model.predict(np.array([u]), np.array([seq]), np.array(item_list))
            rank = predictions[0].argsort().argsort()[0].item()   # .item()方法用于从单元素张量中提取元素值并返回该值，同时保持元素类型不变
            # 可视化
            # 物品：[正例, 负例1, 负例2, 负例3, 负例4]
            # 索引列表：[0, 1, 2, 3, 4]，预测分数：[0.6, 0.9, 0.1, 0.7, 0.5]
            # 正样本的分数排名应该是第3大，按降序排列应该是在放在索引2
            # 添加负号：[-0.6, -0.9, -0.1, -0.7, -0.5]
            # 第一次argsort，根据预测分数，得到输出列表1：[1, 3, 0, 4, 2]
            # 第二次argsort，根据上面的输出列表1，得到输出列表2：[2 0 4 1 3]，第i表示原始位置i在排序中的排名
            # 算一下输出列表2第0个位置为2，表示原始位置0在物品分数的排名是按降序排列中索引2的位置

            total_users += 1

            # @10前10名，索引从0开始
            rank = rank + 1
            if rank <= 10: 
                hit_10 += 1
                dcg = 1 / math.log2(rank + 1)
                idcg = 1 / math.log2(1 + 1)  # 理想情况：排第1名
                ndcg_10 += dcg / idcg

        hit_rate = hit_10 / total_users if total_users > 0 else 0
        ndcg = ndcg_10 / total_users if total_users > 0 else 0

        print('test (HR@10: %.4f, NDCG@10: %.4f)' % (hit_rate, ndcg))
        return hit_rate, ndcg

def data_process(file_name):
    data = pd.read_csv(file_name,
                    sep='::',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                    engine='python')
    
    user_num = 0
    item_num = 0
    User = defaultdict(list)
    # 只划分训练集和测试集，不做验证集，但是训练过程中，一段时间会对测试集进行评测（其实就相当于验证集了哈哈哈）
    user_train = {}
    user_test = {}
    for index, row in data.iterrows():
        if index % 100000 == 0:
            print("已处理 %d 条数据" % index)

        user = int(row['user_id'])
        item = int(row['movie_id'])

        # 因为user_id和movie_id是连续的整数编号，并从1开始，所以可以用最大值来确定总数
        user_num = max(user, user_num)
        item_num = max(item, item_num)

        rating = int(row['rating'])
        timestamp = int(row['timestamp'])
        User[user].append([timestamp, item, rating])
        
    for userid in User.keys():
        # 根据时间轴排序，构建每个用户的历史交互序列
        User[userid].sort(key=lambda x: x[0])

    for userid in User.keys():
        if len(User[userid]) < 4:
            continue
        user_train[userid] = []
        for i_seq in range(len(User[userid])-1):
            user_train[userid].append(User[userid][i_seq][1])
        user_test[userid] = []
        user_test[userid].append(User[userid][-1][1])
    return [user_train, user_test, user_num, item_num]

def get_batch(user_train, item_num, batch_uids, maxlen):
    # batch_uids: [batch_size,]
    seq_list = []
    pos_list = []
    neg_list = []
    for user in batch_uids:
        seq = np.zeros([maxlen], dtype=np.int32)  # 因为item序列都是数字，所以设置np.int32
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1
        cur_user_items = set(user_train[user])  # 记录当前用户交互过的item
        # 从末尾开始填，这样长度小于maxlen的序列左边就能用0填充
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # 负样本从其他用户交互过的item里采样（且不能出现目标用户交互过的item）
            # 因为数据集里item是从1连续，所以可以用这种方法找到某个item，而不需要另外存储所有用户的item的集合，再从里面找
            neg[idx] = np.random.randint(1, item_num + 1)
            while neg[idx] in cur_user_items:
                neg[idx] = np.random.randint(1, item_num + 1)
            nxt = i
            idx -= 1
            if idx == -1: 
                break
        seq_list.append(seq)  # [batch_size, maxlen,]
        pos_list.append(pos)  # [batch_size, maxlen,]
        neg_list.append(neg)  # [batch_size, maxlen,]
    
        # 构造可视化
        # user_train[uid] = [1, 2, 3, 4], maxlen=6
        # seq = [0 0 0 1 2 3]
        # pos = [0 0 0 2 3 4]
        # neg = [0 0 0 6 22 11]
        # 即seq[0:t+1]项的下一个预测item为pos[t]

    return batch_uids, seq_list, pos_list, neg_list

if __name__ == '__main__':
    # 数据准备
    dataset = "./data/MovieLens-1m/ratings.dat"
    dataset_name = 'MovieLens1M'
    [user_train, user_test, user_num, item_num] = data_process(dataset)

    # 训练配置，部分参数得在模型的init里设置好
    model = SASRec(user_num, item_num)
    model.to(model.device)

    # # 测试，先用少量数据能不能跑通
    # test_user_train = dict()
    # test_user_test = dict()
    # for i in range(1,10):  # 使用user_id为1-10的用户
    #     test_user_train[i] = user_train[i]
    #     test_user_test[i] = user_test[i]
    # user_train = test_user_train
    # user_test = test_user_test


    num_batch = (len(user_train)-1) // model.batch_size + 1  # 直接整除会丢失构不成批次大小的数据，比如5//2=2
    print("训练集大小: %d, 批次数: %d" % (len(user_train), num_batch))
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    # 初始化每个层的参数
    for name, param in model.named_parameters():
        # print(name, param.size())
        try:
            # Xavier初始化通过精心设计的方差，为深度学习模型提供了一个"黄金起点"
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    # 不好的初始化会导致：
    # 梯度消失：权重太小，信号在深层网络中衰减
    # 梯度爆炸：权重太大，梯度数值不稳定
    # 训练困难：收敛缓慢或不收敛

    # 将位置嵌入和物品嵌入的第0行置零，因为物品id从1开始，然后位置也从1开始算，所以可以将没用0位来表示填充符
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # 训练或推理
    inference_only = False
    if not inference_only:
        # 训练
        model.train() # enable model training
        epoch_start_idx = 1
        if len(model.state_dict_path)>0:
            try:
                # 加载训练过的模型权重，从断点开始训练
                model.load_state_dict(torch.load(model.state_dict_path, map_location=torch.device(model.device)))
                tail = model.state_dict_path[model.state_dict_path.find('epoch=') + 6:]
                epoch_start_idx = int(tail[:tail.find('.')]) + 1
            except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
                print('failed loading state_dicts, pls check file path: ', end="")
                print(model.state_dict_path)
                print('pdb enabled for your quick check, pls type exit() if you do not need it')
                import pdb; pdb.set_trace()
        # 定义损失函数和优化器
        # 论文用的损失函数是BCE Loss
        bce_criterion = torch.nn.BCELoss()
        # 论文用的优化器是ADAM优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, betas=(0.9, 0.98))
        
        total_time = 0
        start_time = time.time()
        best_test_ndcg, best_test_hr = 0.0, 0.0

        os.makedirs(f"./result/SASRec/{str(start_time).split('.')[0]}")
        f = open(f"./result/SASRec/{str(start_time).split('.')[0]}/log.txt", 'w')

        for epoch in range(epoch_start_idx, model.num_epochs + 1):
            # 每个epoch训练的user需要打乱
            uids = list(user_train.keys())
            random.shuffle(uids)
            for step in range(num_batch):
                start_idx = step * model.batch_size
                end_idx = min((step + 1) * model.batch_size, len(uids))  # 防止最后一批超出范围
                u, seq, pos, neg = get_batch(user_train, item_num, uids[start_idx:end_idx], model.maxlen) 
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg) # list to ndarray，连续内存块，存储原始数值数据，方便后续数值计算操作
                # 前向传播，得到正例和负例的预测logits
                pos_logits, neg_logits = model(u, seq, pos, neg)
                # 创建标签：正例为1，负例为0
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=model.device), torch.zeros(neg_logits.shape, device=model.device)
                # 清空梯度
                optimizer.zero_grad()
                # 找到序列非0开始的部分，从当前索引开始计算损失
                # indices: [2, m]，m为pos里面共batch_size*sequence个数里的非零个数
                # pos是个二维矩阵，所以indices[0][0]存储的是第1个非零数在pos的横坐标，indices[1][0]存储的是第1个非零数在pos的纵坐标
                indices = np.where(pos != 0)
                
                # 论文的损失公式计算
                loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + bce_criterion(neg_logits[indices], neg_labels[indices])
                # for param in model.item_emb.parameters(): 
                    # 对物品嵌入层添加L2正则化(‖w‖₂), 
                    # loss += model.l2_emb * torch.sum(param ** 2)    
                loss.backward()
                optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
            
            if epoch % 20 == 0:
                model.eval()
                total_time = time.time() - start_time
                print("epoch:%d, time: %f(s)"%(epoch, total_time))
                test_hr, test_ndcg = model.evaluate(user_train, user_test, user_num, item_num, inference_only)

                # 保存最优参数
                if test_hr > best_test_hr or test_ndcg > best_test_ndcg:
                    best_test_hr = test_hr
                    best_test_ndcg = test_ndcg
                    fname = 'dataset={}_epoch={}_lr={}_layer={}_head={}_hidden={}_maxlen={}.pth'
                    fname = fname.format(dataset_name, epoch, model.learning_rate, model.num_blocks, model.num_heads, model.embed_dim, model.maxlen)
                    torch.save(model.state_dict(), os.path.join(f"./result/SASRec/{str(start_time).split('.')[0]}/", fname))
                f.write(str(epoch) + '  hr@10:' + str(test_hr) + ' ndcg@10:' + str(test_ndcg) + '\n')
                f.flush()
                model.train()
            
            # 保存最后一个epoch的模型参数
            if epoch == model.num_epochs:
                fname = 'dataset={}_epoch={}_lr={}_layer={}_head={}_hidden={}_maxlen={}.pth'
                fname = fname.format(dataset_name, epoch, model.learning_rate, model.num_blocks, model.num_heads, model.embed_dim, model.maxlen)
                torch.save(model.state_dict(), os.path.join(f"./result/SASRec/{str(start_time).split('.')[0]}/", fname))
                
        f.close()
    else:
        if len(model.state_dict_path)>0:
            try:
                # 加载训练过的模型权重，从断点开始训练
                model.load_state_dict(torch.load(model.state_dict_path, map_location=torch.device(model.device)))
            except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
                print('failed loading state_dicts, pls check file path: ', end="")
                print(model.state_dict_path)
                print('pdb enabled for your quick check, pls type exit() if you do not need it')
                import pdb; pdb.set_trace()
        model.eval()
        test_hr, test_ndcg = model.evaluate(user_train, user_test, user_num, item_num, inference_only)

    print("Done")