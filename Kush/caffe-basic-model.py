#from https://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_mode_cpu()

net = caffe.Net('conv.prototxt', caffe.TEST)

print(net.blobs['conv'].data.shape)

im = np.array(Image.open('examples/images/cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

# Let’s compute the blobs given this input

net.forward()
# Now net.blobs['conv'] is filled with data, and the 3 pictures inside each of the 3 neurons (net.blobs['conv'].data[0,i]) can be plotted easily.

# To save the net parameters net.params, just call :

net.save('mymodel.caffemodel')
