# Ninapro_2023
肌电信号深度学习毕业论文，知网doi：	10.27175/d.cnki.gjxcu.2023.001431
主要实现文件
（1）GetRaw读取.mat文件存储为h5   
（2）EMGFilter信号去噪声 
（3）DFactionSeg 信号分割 标准化 划分数据集  
（4）TrainModel文件下EmgTrain进行训练生成的模型参数会保留 
（5）TestModel文件下的EMGTest进行测试 
（6）Util文件夹txtutil方法将测试结果读取精度存入excel中
其他文件是备份和其他方法的实现，如能量核、12支路，vnt网络，shape方法和特征提取
