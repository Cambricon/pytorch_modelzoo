import sys
import pandas as pd


log1_location=sys.argv[1]
log2_location=sys.argv[2]
epoch=sys.argv[3]

df_cpu=pd.read_csv(log1_location + '/epoch'+str(epoch)+'_rank-1.txt',header=None)
df_mlu=pd.read_csv(log2_location + '/epoch'+str(epoch)+'_rank-1.txt',header=None)

a=df_cpu.loc[:,0]
m=df_mlu.loc[:,0]
corr=df_cpu.corrwith(m,axis=0)
corr_r2=corr[0]**2
print('epoch %s R2: %f' % (epoch, corr_r2))
