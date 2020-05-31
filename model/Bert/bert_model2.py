from utils import CosineLayer 
from keras_bert import load_trained_model_from_checkpoint
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from data2 import train_data_generator
from keras.callbacks import EarlyStopping

class BertConfig():
    #Configuration for BertModel
    def __init__(self,
                 learning_rate=5e-4,
                 min_learning_rate=1e-5,
                 maxlen = 510,
                 config_path=r"./bert_config.json",
                 checkpoint_path='./bert_model.ckpt',
                 dict_path='./vocab.txt',
                 trainable=False):
        '''
        learning_rate:学习率
        min_learning_rate:最低学习率
        maxlen = 510 最长输入文本长度，超过将被截断
        config_path:bert模型配置文件
        checkpoint_path:bert模型数据文件
        dict_path: bert模型字典文件
        trainable: 是否训练bert模型
    
        '''
        self.learning_rate=learning_rate
        self.min_learning_rate=min_learning_rate
        self.maxlen=maxlen
        self.trainable=trainable
        self.config_path=config_path
        self.dict_path=dict_path
        self.checkpoint_path=checkpoint_path
        self.dropout=0.15


class BertModel():
    def __init__(self,config):
        self.config=config
        self.bert_model = load_trained_model_from_checkpoint(config.config_path, config.checkpoint_path, seq_len=None)
        
    def build_model(self):
        self.inputs=[]
        self.outputs=[]
        
        input_x_word=Input(shape=(None,))
        input_mask=Input(shape=(None,))
        output_mask=Input(shape=(None,),dtype="bool")
        
        self.inputs.append(input_x_word)
        self.inputs.append(input_mask)
        self.inputs.append(output_mask)
        
        x = self.bert_model([input_x_word, input_mask])
        bert_out = Lambda(lambda x:tf.boolean_mask(x[0],x[1]))([x,output_mask])

        input_x_word=Input(shape=(None,))
        input_mask=Input(shape=(None,))
        output_mask=Input(shape=(None,),dtype="bool")
        
        self.inputs.append(input_x_word)
        self.inputs.append(input_mask)
        self.inputs.append(output_mask)

        x = self.bert_model([input_x_word, input_mask])
        bert_out1 = Lambda(lambda x:tf.boolean_mask(x[0],x[1]))([x,output_mask])

        c=Lambda(lambda x:K.concatenate([x[0],x[1]] , axis=0))([bert_out,bert_out1])
        
        concat = Lambda(lambda x:tf.reshape(x,[-1,768*3]))(c)
        dense1 = Dense(128, activation='relu')(concat)
        dropout = Dropout(self.config.dropout)(dense1)
        out = Dense(1,activation='sigmoid')(dropout)

        self.outputs.append(out)

        self.model = Model(inputs = self.inputs,outputs = self.outputs)
        self.model.compile(optimizer=Adam(self.config.learning_rate),
                      loss='binary_crossentropy')
        
if __name__=='__main__':
    config=BertConfig()
    model=BertModel(config)
    model.build_model()
    
    data=train_data_generator()
    train=data.train_batch()
    test=data.test_batch()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)

    model.model.fit_generator(
    train,
    steps_per_epoch=data.train_step,
    epochs=50,
    validation_data=test,
    validation_steps=data.test_step,
    callbacks=[early_stopping]
)
