import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Multiply, concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
import argparse

def embedding_model(num_users, num_items, layers=[32,64,32,16], reg_layers=[0]):
    
    num_layer = len(layers)

    MF_embedding_item = Embedding(input_dim = num_items, output_dim = 6)
    MLP_embedding_item = Embedding(input_dim = num_items, output_dim = 6) 
    
    user_input = Input(shape = (6,), dtype='float32', name='user_unput')
    item_input = Input(shape = (1,), dtype='int32', name='item_input')
    
    
    #MF part
    mf_user_latent = (user_input)
    mf_item_latent = Flatten()(MF_embedding_item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])
    mf_vector = (mf_vector)

    print("\n\n", mf_vector.shape)
    #MLP part
    mlp_user_latent = Flatten()(user_input)
    mlp_item_latent = Flatten()(MLP_embedding_item(item_input))
    mlp_vector = concatenate([mlp_user_latent,mlp_item_latent],axis = 1)
    print("\n\n", mlp_vector.shape, "\n\n")
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)
    

    predict_vector = concatenate([mf_vector, mlp_vector])
    prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    #MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    #MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    #MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
    
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis = 0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('predictioin').set_weights([0.5*new_weights, 0.5*new_b])
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [] ,[], []
    num_users = train.shape[0]
    num_items = train.shape[1]
    for(u, i) in train.keys():
        #positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        #negative instances
        for t in range(1,num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels