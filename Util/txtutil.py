import csv
import pandas as pd
import numpy as np
import xlwt
import re

def txt_xls(filename, xlsname):
    start_row = 0
    start_col = 0
    f = open(filename,encoding = 'utf-8')
    xls = xlwt.Workbook(encoding = 'utf-8')
    # 生成excel的方法，声明excel
    sheet = xls.add_sheet('sheet1')
    row_excel = start_row
    for line in f:
        line = line.strip('\n')
        line = line.split()

        print(line)

        col_excel = start_col
        len_line = len(line)
        for j in range(len_line):
            # print(line[j])
            sheet.write(row_excel, col_excel, line[j])
            col_excel += 1
        xls.save(xlsname) # 保存为xls文件

        row_excel += 1
    f.close()

rootdata="D:/Pengxiangdong/ZX/DB2/result/restimulus2/"
#从原始的多个txt文件导入xls
if __name__ == '__main__':

    for j in range(1,41):
        newdata=[]
        data=[]
        filepath=rootdata+"346-1-25/DB2_s"+str(j)+"emgfea.txt"
        with open(filepath,encoding='utf-8',) as txtfile:
            line=txtfile.readlines()
            for i,rows in enumerate(line):
                if i in [52,53] :  #指定数据哪几行
                    data.append(rows)
        txtfile.close()
        newdata=(data[0]+data[1]).replace("\n", "")
        # #写入
        path=rootdata+"346-1Statistics.txt"
        with open(path, "a", ) as f:
            f.writelines(newdata+'\n')
            f.close()
    #写入excel
    xlsname=rootdata+"346-1Statistics.xls"
    txt_xls(path,xlsname)



#从汇总的txt文件导入xls
# if __name__ == '__main__':
#     path="E:/360MoveData/Users/Administrator/Desktop/Statistics.txt"
#     # 写入excel
#     xlsname = "E:/360MoveData/Users/Administrator/Desktop/Statistics.xls"
#     txt_xls(path, xlsname)
