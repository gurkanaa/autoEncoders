#Preprocessing operation
#takes fft of case western data
#but use overlapping long frames for fft
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import scipy.io
from collections import namedtuple
import json

#Data path
#NormalBaseline: for healthy data
baseline_path="/Users/gurkanAydemir/Documents/akademik/PhD/python/virtualenv/CWRU/NormalBaseline/"
#48DriveEndFault
path="/Users/gurkanAydemir/Documents/akademik/PhD/python/virtualenv/CWRU/48DriveEndFault/"
#save path
save_path="/Users/gurkanAydemir/Documents/akademik/PhD/python/virtualenv/autoEncoders/"
#parameters
fft_length=32768
sliding_distance=1000


data_cwru=[]
data_tuple=('Instance_of_data',['Health_status','data','label'])
folders=os.listdir(path)

for folder_id in folders:
    if ('DS' not in folder_id):
        files=os.listdir(path+folder_id)
        for file_id in files:
            if file_id.endswith('.mat'):
                if('Inner' in file_id):
                    health_status='Inner'
                    label=1
                elif('Outer'in file_id):
                    health_status='Outer'
                    label=2
                elif('Ball' in file_id):
                    health_status='Outer'
                    label=2
                #read matfile
                mat_obj=scipy.io.loadmat(path+folder_id+'/'+file_id)
                for key in mat_obj.keys():
                    if 'DE_time' in key:
                        klm=np.asarray(mat_obj[key])
                print(len(klm),int(np.floor((len(klm)-fft_length)/sliding_distance)))
                #computing one sided spectrum
                for i in range(int(np.floor((len(klm)-fft_length)/sliding_distance))):
                        if i==0:
                            xy=np.asarray(klm[i*sliding_distance:i*sliding_distance+fft_length])
                            data=xy
                            Y=np.absolute(np.fft.fft(xy,axis=0))
                            P1=Y[0:int(fft_length/2)+1]
                            P1[1:-1]=2*P1[1:-1]
                            data_fft=P1
                        else:
                            xy=np.asarray(klm[i*sliding_distance:i*sliding_distance+fft_length])
                            Y=np.absolute(np.fft.fft(xy,axis=0))
                            P1=Y[0:int(fft_length/2)+1]
                            P1[1:-1]=2*P1[1:-1]
                            data_fft=np.concatenate((data_fft,P1),axis=1)
                            data=np.concatenate((data,xy),axis=1)
                #save the fft to the same file. 2d array : #of frames*length_of_spectrum
                #np.save(path+folder_id+'/'+file_id[:-4]+'_fft',data_fft)
                print(np.size(data_fft,axis=0))
                print(np.size(data_fft,axis=1))
                xyz=data_tuple(health_status,data_fft,label)
                data_cwru=[data_cwru,xyz]
                del data
                del data_fft
                print(data_cwru)

json = json.dumps(data_cwru)
f = open(save_path+'data_cwru.json',"w")
f.write(json)
f.close()
