import sklearn
import shap
import h5py
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import nina_funcs as nf

from Util.function import get_twoSet

root_data = 'D:/Pengxiangdong/ZX/'
for j in range(1, 2):
    feaFile = h5py.File(root_data + 'DB2/data/stimulus/Fea/DB2_s' + str(j) + 'frefea.h5', 'r')
    # 将六次重复手势分开存储
    fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
    feaFile.close()

    fea_train, fea_test, feay_train, feay_test = get_twoSet(fea_all, fea_label, fea_rep )

    fea_train = fea_train.reshape(fea_train.shape[0], -1, 12)
    fea_test = fea_test.reshape(fea_test.shape[0], -1, 12)

    ch = 2
    temp = fea_train[:, :, ch - 1]
    temp = temp.reshape(temp.shape[0],  -1)

    temp2 = fea_test[:, :, ch - 1]
    temp2 = temp2.reshape(temp2.shape[0],  -1)

    fea_names =["fr", "mnp", "mnf", "mdf", "pkf"]

    # aaa=[]
    # for i in range(5):
    #     aaa.append(list(fea_names[i]))

    # fea_names = ["iemg", "rms", "entropy", "kurtosis", "zero_cross", "mean", "median", "wl",  "ssc", "wamp"]
    fea_names =["fr", "mnp", "mnf", "mdf", "pkf"]
    model = xgb.XGBClassifier(eval_metric='mlogloss',encoder=False).fit(temp, feay_train)
    explainer = shap.TreeExplainer(model)
    expected_value=explainer.expected_value
    shap_values = explainer.shap_values(temp)
    shap_values2 = explainer(temp)
    # shap.plots.bar(shap_values2[1], show_data=False)
    # shap.multioutput_decision_plot(shap_values,temp)
    shap.decision_plot(expected_value[0],shap_values[0][::200,:],features=temp,feature_names=fea_names,highlight=0,ignore_warnings=True)

    # shap.summary_plot(shap_values, temp,
    #                   plot_type="bar",
    #                   feature_names=fea_names,
    #                   max_display=10,
    #                   title="ch"+str(ch),
    #                   show=True)


    # # select a set of background examples to take an expectation over
    # background = temp[np.random.choice(temp.shape[0], 100, replace=False)]
    #
    # # explain predictions of the model on three images
    # e = shap.DeepExplainer(model, background)
    # # ...or pass tensors directly
    # # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    # shap_values = e.shap_values(x_test[1:5])