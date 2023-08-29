#--------------------------Import Moudle--------------------------------------------#
import tensorflow as tf
from tensorflow import keras
import keras.layers as layer
from keras import backend as K
import torch.nn as nn
import torch
#------------------------------------------------------------------------------------#
#--------------------------Definition of Blocks--------------------------------------#
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(ConvBlock,self).__init__()

        self.conv1 = layer.Conv2D(num_filter,kernel_size=3,strides=2, padding= 'valid')
        self.conv2 = layer.Conv2D(num_filter,kernel_size=3,strides=2, padding= 'valid')

        self.batch1 = layer.BatchNormalization()
        self.batch2 = layer.BatchNormalization()

        self.activation = layer.Activation('leaky_relu')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.activation(x)

        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_filter):
        super(EncoderBlock,self).__init__()

        self.conv_block = ConvBlock(num_filter)
        self.pool = layer.MaxPool2D((2,2))

    def call(self, inputs):

            x = self.conv_block(inputs)

            p = self.pool(x)

            return x,p

class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, x, g ):
        super(AttentionGate,self).__init__()
    # take g which is the spatially smaller signal, do a conv to get the same
    # number of feature channels as x (bigger spatially)
    # do a conv on x to also get same geature channels (theta_x)
    # then, upsample g to be same size as x 
    # add x and g (concat_xg)
    # relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
        self.conv_theta_x = nn.Conv2d(x, x, krenel_size = (1,1), stride =(2,2))
        self.conv_phi_g = nn.Conv2d(g, g, kernel_size=(1,1))
        self.att = nn.Sequential(nn.ReLU(), nn.Conv2d(x, 1, kernel_size= (1,1)), nn.Sigmoid(), nn.Upsample(scale_factor=2) )


    def call(self, x, g):
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(g)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)

        return torch.mul(res,x)


class DecoderBloack(tf.keras.layers.Layer):
    def __init__(self, num_filters, information):
        super(DecoderBloack, self).__init__()

        self.attention = AttentionGate(num_filters, information)
        self.up = layer.Conv2DTranspose(num_filters,(2,2),stides =2 , padding= 'valid')
        self.conv_block = ConvBlock(num_filters)
    
    def call(self, inputs, infor):
        infor = self.attention(x, infor)
        x = self.up(inputs)
        x = layer.Concatenate()([x, infor])
        x =self.conv_block(x)

        return x

#--------------------------------------------------------------------------------------#

#--------------------------------Attention-Unet----------------------------------------#
class attention_UNET(tf.keras.Model):
    def __init__(self, n_classes):
        super(attention_UNET, self).__init__()

    # Encoder Block
        self.e1 =EncoderBlock(64)
        self.e2 =EncoderBlock(128)
        self.e3 =EncoderBlock(256)
        self.e4 =EncoderBlock(512)

    # Bridge
        self.b = ConvBlock(1024)
    
    # Decoder
        self.d1 = DecoderBloack(512)
        self.d2 = DecoderBloack(256)
        self.d3 = DecoderBloack(128)
        self.d4 = DecoderBloack(64)
    
    #Output
        # if n_classes == 1:
        #     activation = 'sigmoid'
        # else:
        #     activation = 'softmax'
    
        self.outputs = layer.Conv2D(n_classes,kernel_size=1,padding='valid', activation='sigmoid')

    
    def call(self, inputs):

        inf1, enc1 = self.e1(inputs)
        inf2, enc2 = self.e2(enc1)
        inf3, enc3 = self.e3(enc2)
        inf4, enc4 = self.e4(enc3)

        res = self.b(enc4)

        dec1 = self.d1(res, inf4)
        dec2 = self.d2(dec1, inf3)
        dec3 = self.d3(dec2, inf2)
        dec4 = self.d4(dec3, inf1)

        outputs = self.outputs(dec4)

        model = tf.keras.Model(input = inputs, ouptut = outputs)


        return model
#-------------------------------------------------------------------------------------#

 
# -------------------------------------------------------------------------------------#