推荐系统入门，推荐模型/算法的论文复现

</br>

已复现：ItemCF（召回评估版本+SASRec的负采样评估版本）、SASRec

采用SASRec论文里面的负采样评估

|              | ItemCF                                  | SASRec                            | 论文SASRec                        |
| ------------ | --------------------------------------- | --------------------------------- | --------------------------------- |
| MovieLens-1M | Hit Rate@10: 0.5227</br>NDCG@10: 0.3061 | HR@10: 0.8180</br>NDCG@10: 0.5811 | HR@10: 0.8245</br>NDCG@10: 0.5905 |

SASRec优化：

base：HR@10: 0.8180，NDCG@10: 0.5811

base+post norm：HR@10: 0.8060，NDCG@10: 0.5760

base+(block=4, num_heads=2)：HR@10: 0.8119, NDCG@10: 0.5758

</br>

参考代码：

[pmixer/SASRec.pytorch: PyTorch(1.6+) implementation of https://github.com/kang205/SASRec](https://github.com/pmixer/SASRec.pytorch/tree/main)

[Lockvictor/MovieLens-RecSys: 基于MovieLens-1M数据集实现的协同过滤算法demo](https://github.com/Lockvictor/MovieLens-RecSys)