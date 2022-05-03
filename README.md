# Reimplementation of Relation-Aware Transformer for Portfolio Policy Learning (RAT) (IJCAI 2020)
> Levon 2022.04.29


## Bugs to fix

+ [ ] pandas.panel() 被官方舍弃，我重新通过 multi_index 实现高维表格
+ [ ] 官方仓库提供的 1.pkl 预训练模型如何导入，torch.load() 报错