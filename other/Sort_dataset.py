# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:44:03 2020
@author: zelin
"""
import numpy as np
import h5py

type_names = ['a','E','F','I','j','L','N','R','V']
fdataset = np.zeros((1,340+1))
for i in range(len(type_names)):
    type_name = type_names[i]
    ##### Load data #####
    data_path = 'data/S1_E1_A1.mat'.format(type_name)
    print(data_path)
#     mat = h5py.File(data_path)
#     keys = ''.join(list(mat.keys()))
#     fmat = np.transpose(mat[keys])
#     m_in = fmat.min()
#     m_ax = fmat.max()
#     ffmat = (fmat-m_in)/(m_ax-m_in)
#
#     label_col = np.ones((ffmat.shape[0],1))*(i)
#     class_dataset = np.c_[ffmat,label_col]
#     fdataset = np.r_[fdataset,class_dataset]
#
# np.save('dataset.npy',fdataset[1:,:])
    
    
    
    
    
