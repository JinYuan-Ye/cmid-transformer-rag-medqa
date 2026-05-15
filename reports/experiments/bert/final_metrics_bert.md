# BERT Multi-Task Evaluation

Method A Recall@1: 0.1000
Method C Recall@1: 0.0200

## Cases

--- Case 1 ---
Input: 前段时间发现龟头长了些颗粒，现在发现长得更多了，会痒，没有乱搞过，怀疑得了尖锐湿疣，有什么办法能治好？
GT Std: 查询 头 的具体操作步骤及实施方法。
Method A: 查询 ， 的相关医疗信息。 (Score: 0.4066)
Method C (Rewrite): 查 询 头 的 标 准 治 疗 方 案 及 手 段 。
Method C (Retrieval): 查询药物 手 的适应症及适用人群。 (Score: 0.3338)
--- Case 2 ---
Input: 患有内痔七八年，干活多了或上火了就较严重,肛门内边会出现有大豆大的疙瘩,劳累后就会痛,有什么偏方可根除？
GT Std: 查询 有内 的标准治疗方案及手段。
Method A: 查询关于 ,肛 的综合医疗信息解答。 (Score: 0.3543)
Method C (Rewrite): 查 询 肛 门 的 标 准 治 疗 方 案 及 手 段 。
Method C (Retrieval): 查询药物 手 的适应症及适用人群。 (Score: 0.3329)
--- Case 3 ---
Input: 没有病史，这是第一次，医生说长了湿疣，在阴道上也长了，这样的情况下怎么治疗？
GT Std: 查询 阴道 的标准治疗方案及手段。
Method A: 查询 ，下 的标准治疗方案及手段。 (Score: 0.4472)
Method C (Rewrite): 查 询 阴 道 的 标 准 治 疗 方 案 及 手 段 。
Method C (Retrieval): 查询 ，阴 的相关医疗信息。 (Score: 0.3289)