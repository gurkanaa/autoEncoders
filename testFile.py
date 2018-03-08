import autoencoderKeras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
data=autoencoderKeras.load_data('data_cwru_small.txt')
tr_data_array,test_data_array,tr_data_label,test_data_label=autoencoderKeras.cwru_tra_test_sets(data)
model=autoencoderKeras.create_model(3,16385,2048,512,128)
print(len(model.get_weights()))
model=autoencoderKeras.pretrain(model,np.transpose(tr_data_array),0.1)
model=autoencoderKeras.overall_train(model,np.transpose(tr_data_array),0.1)
test = model.predict(np.transpose(test_data_array))
sio.savemat('test.mat', {'test':test})
training=model.predict(np.transpose(tr_data_array))
sio.savemat('training.mat', {'training':training})
