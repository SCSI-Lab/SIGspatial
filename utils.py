import PIL as Image
import numpy as np

def image_to_array(filenames, size, channel):
    '''
    Converting RGB images to numpy arrays.
    
    Input
    ----------
        filenames: an iterable of the path of image files
        size: the output size (height == width) of image. 
              Processed through PIL.Image.NEAREST
        channel: number of image channels, e.g. channel=3 for RGB.
        
    Output
    ----------
        An array with shape = (filenum, size, size, channel)
        
    '''
    # number of filesP
    L = len(filenames)
    
    # allocation
    out = np.empty((L, size, size, channel))
    
    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ..., 0] = np.array(pix)
    else:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]