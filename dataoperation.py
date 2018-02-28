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
import json
#Data path
#NormalBaseline: for healthy data
baseline_path="/home/gurkan/Desktop/mlann/CWRU/NormalBaseline/"
#48DriveEndFault
fault_path="/home/gurkan/Desktop/mlann/CWRU/48DriveEndFault/"
#save path
save_path="/home/gurkan/Desktop/mlann/CWRU/"
#parameters
fft_length=32768
sliding_distance=1000
data_cwru=[]
tuple_id=0
for fault in range(1):
    if fault==0:
        path=baseline_path
    else:
        path=fault_path
    folders=os.listdir(path)
    for folder_id in folders:
        if ('DS' not in folder_id):
            files=os.listdir(path+folder_id)
            for file_id in files:
                if file_id.endswith('.mat'):
                    if('Normal' in file_id):
                        health_status='Normal'
                        label=1
                    elif('Outer'in file_id):
                        health_status='Outer'
                        label=2
                    elif('Ball' in file_id):
                        health_status='Ball'
                        label=3
                    elif('Inner' in file_id):
                        health_status='Inner'
                        label=4
                    #read matfile
                    mat_obj=scipy.io.loadmat(path+folder_id+'/'+file_id)
                    for key in mat_obj.keys():
                        if 'DE_time' in key:
                            klm=np.asarray(mat_obj[key])
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
                    data_dict={'Health_status':health_status,'Data':data_fft.tolist(),'label':label}
                    #save the fft to the same file. 2d array : #of frames*length_of_spectrum
                    #np.save(path+folder_id+'/'+file_id[:-4]+'_fft',data_fft)
                    data_cwru.append(data_dict)
                    del data
                    del data_fft
                    print(folder_id+'/'+file_id,data_cwru[tuple_id]['Health_status'],data_cwru[tuple_id]['label'],np.size(data_cwru[tuple_id]['Data'],axis=0),np.size(data_cwru[tuple_id]['Data'],axis=1))
                    tuple_id=tuple_id+1
with open('normal.txt', 'w') as outfile:
    json.dump(data_cwru, outfile)
