import numpy as np
import cv2
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
from test_caffe import evaluate
import pickle
ID = "4"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= ID

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print(l);
            print ('input + output channels need to be the same')
            raise
        if h != w:
            print ('filters need to be square')
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
model = caffe.Net('deploy2.prototxt',caffe.TRAIN)
model.copy_from('vgg16.caffemodel')
# init
#caffe.set_mode_gpu()


#solver = caffe.SGDSolver('solver.prototxt')
#print('done init for solver')
#v do net surgery to set the deconvolution weights for bilinear interpolation

interp_layers = [k for k in model.params.keys() if 'up' in k]
loss_layers = [k for k in model.blobs.keys() if 'loss' in k]
print('done for 2')
interp_surgery(model, interp_layers)

img = cv2.imread('test.jpg')
gth = cv2.imread('test.png',0)
tem1 = img.transpose(2,0,1)
tem1 = tem1/255
tem1 = tem1.reshape(1,*tem1.shape)
tem2 = gth.reshape(1,1,*gth.shape)

model.blobs['data'].reshape(*tem1.shape)# 这里的reshape函数是caffe自带的reshape函数，用来对初始化尺寸的
model.blobs['label'].reshape(*tem2.shape)
model.blobs['data'].data[...] = tem1#这里的data才是真的数据。
model.blobs['label'].data[...] = tem2
model.forward()
param_size = model.params['conv1_2'][0].data.shape
diff_comput = np.zeros(param_size)#the diff computed by net
loss_original = np.zeros(len(loss_layers))
loss_add_epsion = np.zeros(len(loss_layers))#the loss added the epsion
loss_minus_epsion = np.zeros(len(loss_layers))#the loss minus the epsion
original_param = np.zeros(param_size)
epsion = 1e-6
for n,i in enumerate(loss_layers):
    loss_original[n] += model.blobs[i].data
model.backward()
diff_comput += model.params['conv1_2'][0].diff
original_param += model.params['conv1_2'][0].data

model.params['conv1_2'][0].data[45,45,1,1] = model.params['conv1_2'][0].data[45,45,1,1]+epsion
model.forward()
for n,i in enumerate(loss_layers):
    loss_add_epsion[n] += model.blobs[i].data

#epsion = 1e-6
model.params['conv1_2'][0].data[45,45,1,1] = model.params['conv1_2'][0].data[45,45,1,1]-2*epsion
model.forward()
for n,i in enumerate(loss_layers):
    loss_minus_epsion[n] += model.blobs[i].data

grad_test = (loss_add_epsion-loss_minus_epsion)/(2*epsion)
print(grad_test)
print(diff_comput[45,45,1,1])
