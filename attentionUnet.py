from myUnet import EncoderBlock
from myUnet import DecoderBloack
from myUnet import AttentionGate
from myUnet import ConvBlock
import keras.layers as layer

#myUnet의 ConvBlock = Double Convolution
#myUnet의 EncoderBlock = ConvBlock + Maxpool
#myUnet의 DecoderBlock = Attention gate + Up sampling + Concatenate + ConvoBlock
#SE layer와 ASPP block을 추가하여 모델 재구성(Remote Sensing논문)


def att_Unet(inputImage):
    #Encoder Block
    #Later , Add SE layer
    #res는 다음 단계로 넘어갈 matrix
    #inf는 반대편 Attention Unit Layer로 넘겨줄 information
    inf1 , res1 = EncoderBlock.call(inputImage) # 
    inf2 , res2 = EncoderBlock.call(res1) # 
    inf3 , res3 = EncoderBlock.call(res2)
    inf4 , res4 = EncoderBlock.call(res3)

    #Convolution Block (bridge)
    #Later , Add ASPP block
    res = ConvBlock.call(res4)
    
    #Decoder Block
    dec1 = DecoderBloack(res, inf4)
    dec2 = DecoderBloack(dec1, inf3)
    dec3 = DecoderBloack(dec2, inf2)
    dec4 = DecoderBloack(dec3, inf1)

    #Output

    outputs = layer.Conv2D(dec4, kernel_size=1,padding = 'valid', activation = 'sigmoid')

    return outputs


