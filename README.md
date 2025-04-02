# Ninapro_2023
肌电信号深度学习毕业论文，知网：基于多通道特征融合网络的肌电信号手势识别方法研究
主要实现文件
（1）GetRaw读取.mat文件存储为h5   
（2）EMGFilter信号去噪声 
（3）DFactionSeg 对原始信号分割 标准化 划分数据集  
（3-2）GetFeature从时间窗信号中提取特征信号，提取哪些特征可以用shap方法进行分析后替换（此步骤为输入是与时域信号和特征信号的双流网络才需要）
（4）TrainModel文件下EmgTrain进行训练生成的模型参数会保留 
（5）TestModel文件下的EMGTest进行测试 
（6）Util文件夹txtutil方法将测试结果读取精度存入excel中
其他文件是备份和其他方法的实现，如能量核、12支路，vnt网络，shap方法和特征提取
