import keras.backend as K
from keras.layers import Lambda
from keras_bert import Tokenizer


class CosineLayer():
 
    def __call__(self, x1, x2):
 
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_
 
        output_shape = (1,)
        value = Lambda(_cosine,output_shape=output_shape)([x1, x2])
        return value

class OurTokenizer(Tokenizer): #为保证tokenize操作后长度与原先比只加2
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
