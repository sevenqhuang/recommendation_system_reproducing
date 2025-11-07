import math
import random
import pandas as pd
import numpy as np
from collections import defaultdict

random.seed(42)
np.random.seed(42)

class ItemCF():
    def __init__(self):
        self.trainset = {}
        self.testset = {}
        self.movie_popular = {}
        self.movie_similarity = {}   # （基于所有用户，得到物品item与item的相似度
        self.itemsim = {}
        self.all_items = set()  # 所有物品的集合

        self.recommend_num = 10  # 推荐Top-N个物品

    def get_dataset(self):
        # 读取数据集，itemCF只用到user_id、movie_id、rating三列
        data = pd.read_csv('./data/MovieLens-1m/ratings.dat',
                           sep='::',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'],
                           engine='python')
        
        # 首先收集所有物品
        for m_id in data['movie_id']:
            self.all_items.add(int(m_id))
        print(f"总物品数: {len(self.all_items)}")
        
        User = defaultdict(list)
        for index, row in data.iterrows():
            
            if index % 100000 == 0:
                print("已处理 %d 条数据" % index)
            user = int(row['user_id'])
            item = int(row['movie_id'])
            rating = int(row['rating'])
            timestamp = int(row['timestamp'])
            User[user].append([timestamp, item, rating])
        
        for userid in User.keys():
            # 根据时间轴排序，构建每个用户的历史交互序列
            self.trainset.setdefault(userid, {})
            self.testset.setdefault(userid, {})
            User[userid].sort(key=lambda x: x[0])

        for userid in User.keys():
            if len(User[userid]) < 4:
                # 如果用户的交互少于4个，则不进行训练和测试，SASRec复现代码里也是
                continue
            # 序列除最后一个交互外，其余作为训练集
            for i in range(len(User[userid])-1):
                item_i = User[userid][i][1]
                rating = User[userid][i][2]
                self.trainset[userid][item_i] = rating
            
            # 取最后一个交互作为需要预测的测试样本
            item_test = User[userid][-1][1]
            rating_test = User[userid][-1][2]
            self.testset[userid][item_test] = rating_test
    
    def cal_similarity(self):
        for user, items in self.trainset.items():
            for item in items:
                if item not in self.movie_popular:
                    self.movie_popular[item] = 0
                # 喜欢item的用户数
                self.movie_popular[item] += 1
        
        for user, items in self.trainset.items():
            for item_i in items:
                for item_j in items:
                    self.movie_similarity.setdefault(item_i, defaultdict(int))
                    if item_i == item_j:
                        continue
                    # 计算共同喜欢item_i和item_j的用户数，公式的分子部分
                    self.movie_similarity[item_i][item_j] += 1

        for item_i, related_items in self.movie_similarity.items():
            for item_j, cij in related_items.items():
                # 计算相似度，公式的分母部分
                self.movie_similarity[item_i][item_j] = cij / math.sqrt(self.movie_popular[item_i] * self.movie_popular[item_j])
    
    def predict_score(self, user, item):
        """计算用户对指定物品的预测评分"""
        score = 0.0
        interacted_items = self.trainset.get(user, {})
        
        for item_i, rating in interacted_items.items():
            if item_i in self.movie_similarity and item in self.movie_similarity[item_i]:
                similarity = self.movie_similarity[item_i][item]
                score += similarity * rating
        return score

    def evaluate_with_negative_sampling(self, num_negatives=100):
        """使用负采样策略进行评估"""        
        hit_10 = 0
        ndcg_10 = 0
        total_users = 0
        
        # 评估时只考虑训练集中出现的用户，因为只有这些用户我们才有训练数据来为他们生成推荐
        for i, user in enumerate(self.trainset):
            if i % 1000 == 0:
                print(f"已处理 {i} 个用户")

            if user not in self.testset:
                continue
            
            # 获取用户的测试物品（正样本）
            test_items = self.testset[user]
            if len(test_items) == 0:
                continue

            total_users += 1
                
            # 为每个测试物品创建一个评估实例
            for pos_item in test_items:
                # 采样负样本
                negative_candidates = list(self.all_items - set(self.trainset.get(user, {}).keys()) - set(self.testset.get(user, {}).keys()))
                
                if len(negative_candidates) < num_negatives:
                    neg_items = negative_candidates
                else:
                    neg_items = random.sample(negative_candidates, num_negatives)
                
                # 构建候选集：1个正样本 + num_negatives个负样本
                candidate_items = [pos_item] + neg_items
                
                # 计算每个候选物品与用户前面交互过的item的相似度评分
                item_scores = []
                for item in candidate_items:
                    score = self.predict_score(user, item)
                    item_scores.append((item, score))
                
                # 按评分排序
                ranked_items = sorted(item_scores, key=lambda x: x[1], reverse=True)
                
                # 找到正样本的排名
                pos_rank = None
                for rank, (item, score) in enumerate(ranked_items):
                    if item == pos_item:
                        pos_rank = rank + 1  # 排名从1开始
                        break
                
                # 计算Hit@10
                if pos_rank is not None and pos_rank <= 10:
                    hit_10 += 1
                
                # 计算NDCG@10
                if pos_rank is not None and pos_rank <= 10:
                    dcg = 1 / math.log2(pos_rank + 1)
                    idcg = 1 / math.log2(1 + 1)  # 理想情况：排第1名
                    ndcg_10 += dcg / idcg
        
        # 计算平均指标
        hit_rate = hit_10 / total_users if total_users > 0 else 0
        ndcg = ndcg_10 / total_users if total_users > 0 else 0
        
        print(f"Hit Rate@10: {hit_rate:.4f}")
        print(f"NDCG@10: {ndcg:.4f}")
        
        return hit_rate, ndcg

if __name__ == '__main__':
    itemcf = ItemCF()
    itemcf.get_dataset()
    print("训练集样本数：%d" % len(itemcf.trainset))
    print("测试集样本数：%d\n" % len(itemcf.testset))

    itemcf.cal_similarity()
    print("物品相似度计算完成！")
    
    # SASRec负采样评估方法
    print("\n=== 使用负采样评估 ===")
    hr_new, ndcg_new = itemcf.evaluate_with_negative_sampling(num_negatives=100)
    
    