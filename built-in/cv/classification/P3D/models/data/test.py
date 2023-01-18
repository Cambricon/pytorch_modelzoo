
f = open("/algo/algo/yangliu/modelzoo_training/pytorch/pseudo-3d-pytorch/data/ucfTrainTestlist/1.txt","r")
line = f.readlines()
# print(line)
for i in line:
    # line[i]= '/workspace/UCF101/videos/' + line[i]
    i = i.split('/')
    print(i[1])
f.close()
# w = open("/algo/algo/yangliu/modelzoo_training/pytorch/pseudo-3d-pytorch/data/ucfTrainTestlist/1.txt","w")
# w.writelines(line)
    # print(line[0])

# import os
# import shutil
# src = 'ucfTrainTestlist/1.txt'
# dst = '1.txt'
# shutil.copy(src, dst)

