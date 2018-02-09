#takes fft of case western data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
#read data
#macbook
#spectrum_path="/Users/gurkanAydemir/Documents/akademik/PhD/python/autoencoders/NormalBaseline"
#labPC
#spectrum_path='/home/gurkan/Desktop/casedata/data/inner/'
spectrum_path="/home/gurkan/Desktop/casedata/data/normal/"
#parameters
fft_length=2048

files=os.listdir(spectrum_path)
for file_id in files:
    mat_obj=scipy.io.loadmat(spectrum_path+file_id)
    for key in mat_obj.keys():
        if 'DE_time' in key:
            klm=np.asarray(mat_obj[key])
    for i in range(int(np.floor(len(klm)/fft_length))):
        try:
            data
        except NameError:
            xy=np.asarray(klm[i*fft_length:(i+1)*fft_length])
            data=xy
            Y=np.absolute(np.fft.fft(xy,axis=0))
            P1=Y[0:int(fft_length/2)+1]
            P1[1:-1]=2*P1[1:-1]
            data_fft=P1
        else:
            xy=np.asarray(klm[i*fft_length:(i+1)*fft_length])
            Y=np.absolute(np.fft.fft(xy,axis=0))
            P1=Y[0:int(fft_length/2)+1]
            P1[1:-1]=2*P1[1:-1]
            data_fft=np.concatenate((data_fft,P1),axis=1)
            data=np.concatenate((data,xy),axis=1)
    print(np.size(data,axis=1))

np.save(spectrum_path+'fft_all_normal',data_fft)

asdas
asdsa
