#Modules
from keras.models import Model#type of the ANN model
from keras.layers import Dense,Input#fully connected layer
from keras.models import Sequential
from keras import losses
from keras import optimizers
from keras import backend as K
from keras import regularizers
import numpy as np

#create autoencoder model
def create_model(num_of_layers,input_size,num_of_neurons,layer_activations,output_activation):
    for i in range(len(num_of_neurons)):
        if type(num_of_neurons[i])!=int:
            raise Exception('Number of neurons in hidden layers must be integer')
    if len(num_of_neurons)!=num_of_layers and len(num_of_neurons)!=0:
        raise Exception('Number of layers is ',num_of_layers)
    else:
        if len(num_of_neurons)==num_of_layers:
            model_param=num_of_neurons
        else:
            model_param=np.zeros(num_of_layers)
            for i in range(num_of_layers):
                model_param[i]=int(input_size/pow(4,i+1))
        model=Sequential()
        model_param=[int(x) for x in model_param]
        print(model_param)
        #encoder
        for i in range(num_of_layers):
            if i==0:
                model.add(Dense(model_param[i],activation=layer_activations,input_dim=input_size))
            else:
                model.add(Dense(model_param[i],activation=layer_activations))
        #decoder
        for i in range(0,num_of_layers):
            if i==num_of_layers-1:
                model.add(Dense(input_size,activation=output_activation))
            else:
                model.add(Dense(model_param[num_of_layers-i-2],activation=layer_activations))
    return model
#Pretraining of stacked autoencoder
def pretrain(model,data,learning_rate,layer_activations,output_activation,optimizer):
    weights=model.get_weights()
    weigts_dummy=weights
    layer_input=data
    for i in range(int(len(weights)/4)):
        if i==0:
            inputs=Input(shape=(np.size(weights[2*i],axis=0),))
            layer_1=Dense(np.size(weights[2*i],axis=1),activation=layer_activations)(inputs)
            predictions=Dense(np.size(weights[2*i],axis=0),activation=output_activation)(layer_1)
        else:
            inputs=Input(shape=(np.size(weights[2*i],axis=0),))
            layer_1=Dense(np.size(weights[2*i],axis=1),activation=layer_activations)(inputs)
            predictions=Dense(np.size(weights[2*i],axis=0),activation=layer_activations)(layer_1)
        stacked_ae=Model(inputs=inputs,outputs=predictions)
        stacked_ae.compile(optimizer=optimizer, loss='mean_squared_error')
        print(i)
        stacked_ae.fit(layer_input,layer_input,shuffle=True,epochs=10,batch_size=64)
        get_h_layer_output = K.function([stacked_ae.layers[0].input],
                                          [stacked_ae.layers[1].output])
        layer_1_predictions = get_h_layer_output([layer_input, 0])[0]
        w=stacked_ae.get_weights()
        del layer_input,layer_1,inputs,predictions,get_h_layer_output
        layer_input=layer_1_predictions
        weigts_dummy[2*i]=w[0]
        weigts_dummy[2*i+1]=w[1]
        weigts_dummy[len(weights)-2*i-2]=w[2]
        weigts_dummy[len(weights)-2*i-1]=w[3]
        del w
    model.set_weights(weigts_dummy)
    return model
#fine tuning of autoencoder
def overall_train(model,data,learning_rate,optimizer):
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(data,data,shuffle=True,epochs=20,batch_size=64)
    return model
