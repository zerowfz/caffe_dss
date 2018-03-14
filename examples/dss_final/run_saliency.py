import numpy as np
import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
from test_caffe import evaluate
import pickle
ID = "1"
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

base_weights = 'vgg16.caffemodel'  # the vgg16 model

# init
caffe.set_mode_gpu()


solver = caffe.SGDSolver('solver.prototxt')
print('done init for solver')
# do net surgery to set the deconvolution weights for bilinear interpolation

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
print('done for 2')
interp_surgery(solver.net, interp_layers)
print('done for 3')

# copy base weights for fine-tuning
#solver.restore('./snapshot/ours_iter_4000.solverstate')
solver.net.copy_from(base_weights)
print('done for 4')
sal_loss_layer = [k for k in solver.net.blobs.keys() if 'loss' in k and 'edg' not in k]
edge_loss_layer = [k for k in solver.net.blobs.keys() if 'loss' and 'edg' in k]
result = []
PRINT_INTERVAL = 2000
MAX_ITER = 200000
sal = np.zeros([len(sal_loss_layer),MAX_ITER])
sal_iter_b  = 0
sal_iter_a = 0
iter_size_test = 500
iter_edge = 0
solver.net.clear_param_diffs()
test_net = caffe.Net('multi-2.prototxt','./vgg16.caffemodel',caffe.TEST)
output_dir = './result/multi2/2/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

with open('solver.prototxt','r') as f:
    with open(output_dir+'readme','w') as f2:
        for i in f.readlines():
            f2.write(i)
for it in range(1,MAX_ITER+1):
    solver.step(1)
    num = solver.net.blobs['data'].data[0][0].flatten().size
    if solver.net.task == 1:
        for n,l in enumerate(sal_loss_layer):
            sal[n,sal_iter_b] = solver.net.blobs[l].data/num
        sal_iter_b = sal_iter_b+1
    weight = (1-it/MAX_ITER)
    if solver.net.task==1:
        for i in edge_loss_layer:
            solver.net.blobs[i].diff[...] = weight
    print('sal_conv3_dsn1:',np.abs(solver.net.params['sal-conv3-dsn1'][0].diff).mean())
    print('conv1_2:',np.abs(solver.net.params['conv1_2'][0].diff).mean())
    print('edge_conv3_dsn1:',np.abs(solver.net.params['edge-conv3-dsn1'][0].diff).mean())
    assert solver.net.params['edge-conv3-dsn1'][0].diff.flatten().mean()<1e8
    assert np.abs(solver.net.blobs['conv2_1'].data).mean()<1e8
    if it % PRINT_INTERVAL ==0:
        print('Validating...')
        c_mean_loss = sal[:,sal_iter_a:sal_iter_b].mean(1)
        sal_iter_a = sal_iter_b
        solver.net.save(output_dir+'tmp.caffemodel')
        maxF,meanF,mae  = evaluate(test_net,ID,iter_size_test,output_dir)
        print('maxF:',maxF,',meanF:',meanF,',mae:',mae)
        print('loss:',c_mean_loss.mean())
        with open(output_dir+'result.txt','a+') as f:
            f.write(str([it,maxF,meanF,mae,c_mean_loss.mean()])+'\n')
        result.append([it,maxF,meanF,mae,c_mean_loss])
#with open('./result/result.txt','w') as f:
#    f.write(str(result))
with open(output_dir+'result.pickle','wb') as f:
    pickle.dump({'result':result},f,pickle.HIGHEST_PROTOCOL)
#solver.step(140000)

