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
        self.movie_similarity = {}   # （基于所有用户，得到物品item与item的相似度）
        self.itemsim = {}

        self.K = 50 # 取前K个相似度最高的计算相似度矩阵
        self.recommend_num = 10  # 推荐Top-N个物品

    def get_dataset(self, privot=0.8):
        # 读取数据集，itemCF只用到user_id、movie_id、rating三列
        data = pd.read_csv('./data/MovieLens-1m/ratings.dat',
                           sep='::',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'],
                           engine='python')
        trainset_len = 0 
        testset_len = 0
        
        for index, row in data.iterrows():
            if index % 100000 == 0:
                print("已处理 %d 条数据" % index)
            user = int(row['user_id'])
            item = int(row['movie_id'])
            rating = int(row['rating'])
            if random.random() < privot:
                self.trainset.setdefault(user, {})
                # 建立用户-物品的mapping表
                self.trainset[user][item] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                # 建立用户-物品的mapping表
                self.testset[user][item] = int(rating)
                testset_len += 1
        print("trainset 样本数：%d" % trainset_len)
        print("testset 样本数：%d" % testset_len)
    
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
    
    # 生成每个用户的推荐列表
    def recommend(self, user, N=10, K=100):
        rank = {}

        # 得到用户user历史交互过的物品
        interacted_items = self.trainset.get(user, {})
        # 计算用户u对历史中未交互过的物品中的物品j的兴趣
        for item_i, rating in interacted_items.items():
            # self.movie_similarity.get(item_i, {}).items()，获取与item_i相似的所有物品及相似度
            # 相似度从高到低排序，取前K个
            for item_j, similarity in sorted(self.movie_similarity.get(item_i, {}).items(),
                                             key=lambda x: x[1], reverse=True)[:K]:
                # 过滤掉用户已经交互过的物品
                if item_j in interacted_items:
                    continue
                rank.setdefault(item_j, 0)
                rank[item_j] += similarity*rating
        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N]

    def evaluate_original(self):
        """保留原有的评估方法用于对比"""
        # 使用Hit Rate和NDCG评估推荐效果
        # Hit Rate@10 and NDCG@10，N=10
        def getDCG(scores):
            # 公式编码为np.power(2, scores) - 1，其实就是计算所有rel_i=1的位置的值再求和，因为scores=0或1
            # 数组下表从0开始，但是公式中物品位置从1开始，且还要+1防止除0，所以+2
            scores = np.array(scores)
            return np.sum(
                np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                dtype=np.float32)

        hit = 0
        total = 0   # 有效用户计数
        ndcg = 0
        # 评估时只考虑训练集中出现的用户，因为只有这些用户我们才有训练数据来为他们生成推荐
        for i, user in enumerate(self.trainset):
            if i % 1000 == 0:
                print(f"已处理 {i} 个用户")
            if user not in self.testset:
                continue
            
            test_items = self.testset[user]
            if len(test_items) == 0:
                continue

            total += 1

            recommended_items = self.recommend(user, N=self.recommend_num, K=self.K)
            for item, score in recommended_items:
                if item in test_items:
                    # hit分子为命中数，只要命中一个就算命中
                    hit += 1
                    break
            
            dcg = 0
            idcg = 0
            # 计算DCG，获取推荐列表的物品在测试集中是否命中
            # rank_scores记录推荐列表中每个物品是否命中，命中为1，否则为0，对应公式中的rel_i
            rank_scores = np.zeros(self.recommend_num)
            for i, (item, score) in enumerate(recommended_items):
                if item in test_items:
                    rank_scores[i] = 1
                else:
                    rank_scores[i] = 0
            dcg = getDCG(rank_scores)

            # 计算IDCG - 理想情况下的排序
            i_rank_scores = sorted(rank_scores, reverse=True)
            idcg = getDCG(i_rank_scores)
            if idcg == 0:
                continue
            ndcg += dcg / idcg

        hr = hit / total if total != 0 else 0
        ndcg = ndcg / total if total != 0 else 0  # 归一化所有用户的ndcg
    
        print(f"【原方法】Hit Rate@{self.recommend_num}: {hr:.4f}")
        print(f"【原方法】NDCG@{self.recommend_num}: {ndcg:.4f}")

        return hr, ndcg

if __name__ == '__main__':
    itemcf = ItemCF()
    itemcf.get_dataset(0.7)
    print("训练集样本数：%d" % len(itemcf.trainset))
    print("测试集样本数：%d" % len(itemcf.testset))

    itemcf.cal_similarity()
    print("物品相似度计算完成！")

    # 评估方法用于对比
    print("\n=== 原有评估方法 ===")
    hr_old, ndcg_old = itemcf.evaluate_original()
    