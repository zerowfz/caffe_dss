import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import pydensecrf.densecrf as dcrf
import time
#out_root = '../original_dss/caffe_dss/examples/dss_final/'
caffe_root  = '../../'
#caffe_root = '../caffe_dss-master/'
#code_root  = caffe_root+'examples/dss_final/'
code_root = "./"
sys.path.insert(0,caffe_root+'python')
import caffe
import os 
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#params needed to update
#snap_root = code_root +'snapshot/'
#snap_root = code_root + 'output/'
snap_root = 'result/multi2/2/'
#caffemodel_name = 'rest/multi-1-new-edgeloss-160000.caffemodel'
#caffemodel_name = [f'dss_iter_{k}.caffemodel' for k in [100000,120000,140000,160000]]
caffemodel_name = 'tmp.caffemodel'
#caffemodel_name = 'multi-2-120000.caffemodel'
#caffemodel_name = 'dss_iter_120000.caffemodel'
#prototxt_name = 'output2/original.prototxt'
#prototxt_name = 'multi-2.prototxt'
prototxt_name = 'multi-2-1.prototxt'
#prototxt_name = 'multi-2.prototxt'
#output_name = [out_root+f'output_new/multi-1-{i}.pickle' for i in [100000,120000,140000,160000]]
model_name = 'original'
#output_name = out_root + 'output_new2/original.pickle'
iters = 1
interval = 40000
thresholds = 256
E = 1e-4
EPSION = 1e-8
def untransform(img):
    img = img.astype(np.float32)
    img = img.transpose(1,2,0)
    img += np.array([104.00699,116.66877,122.67892])
    img = img[:,:,::-1]
    return img.astype(np.uint8)

def compute_pr(sal,lbl,pre,rec):
    l = lbl>0
    for i in range(thresholds):
        p = sal>i
        ab = p[l].sum()
        a = p.sum()
        b = l.sum()
        pre[i] += (ab+E)/(a+E)
        rec[i] += (ab+E)/(b+E)

def compute_mae(sal,lbl):
    return np.abs(sal-lbl).sum()/sal.size

def sigmoid(x):
    return 1/(1+np.exp(-x))

def crf_compute(img,ann):
    tau = 1.05
    img = img.copy(order='C')
    d = dcrf.DenseCRF2D(img.shape[1],img.shape[0],2)
    U = np.zeros((2,img.shape[0]*img.shape[1]),dtype=np.float32)
    U[1,:] = (-np.log(ann+EPSION)/(tau*sigmoid(ann))).flatten()
    U[0,:] = (-np.log(1-ann+EPSION)/(tau*sigmoid(1-ann))).flatten()
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=5,compat=3)
    d.addPairwiseBilateral(sxy=60,srgb=8,rgbim=img,compat=3)
    infer = np.array(d.inference(5)).astype(np.float32)
    return infer[1].reshape(img.shape[:2])

caffe.set_mode_gpu()
'''
for x in range(1):
    #iters = x*interval+interval
    net = caffe.Net(code_root+prototxt_name,snap_root+caffemodel_name,caffe.TEST)
    #net = caffe.Net(code_root+prototxt_name,snap_root+,caffe.TEST)
    pre = np.zeros([8,256])
    rec = np.zeros([8,256])
    mae = 0
    time1 = []
    time2 = []
    for i in range(1000):
        print('num:',i)
        time_1 = time.time()
        net.forward()
        time1.append(time.time()-time_1)
        out = []
        out.append(net.blobs['sigmoid-fuse'].data[0][0])
        out.append(net.blobs['sigmoid-dsn6'].data[0][0])
        out.append(net.blobs['sigmoid-dsn5'].data[0][0])
        out.append(net.blobs['sigmoid-dsn4'].data[0][0])
        out.append(net.blobs['sigmoid-dsn3'].data[0][0])
        out.append(net.blobs['sigmoid-dsn2'].data[0][0])
        out.append(net.blobs['sigmoid-dsn1'].data[0][0])
        fin = net.blobs['sigmoid-fuse'].data[0][0].copy()
        for m in range(3):
            fin += net.blobs[f'sigmoid-dsn{m+2}'].data[0][0]
        fin = fin/4
        out.append(fin)
        label = net.blobs['label'].data[0][0]
        mae += compute_mae(fin,label)
        #print(mae)
        #img = untransform(net.blobs['data'].data[0])
        for j,fin2 in enumerate(out):
            fin4 = fin2 *255
            compute_pr(fin4,net.blobs['label'].data[0][0],pre[j,:],rec[j,:])
            #time_2 = time.time()
            #fin3 = crf_compute(img,fin2)
            #time2.append(time.time()-time_2)
            #mae += compute_mae(fin3,label)
            #fin3 =fin3 * 255
            #compute_pr(fin3,net.blobs['label'].data[0][0],pre[j+1,:],rec[j+1,:])
            #print('xxxxx:',j,'xxxx:',pre[j,255])
    pre = pre/1000
    rec = rec/1000
    #mae = mae/1000
    time_1 = np.array(time1).mean()
    time_2 = np.array(time2).mean()
    F = np.max(pre*rec*(1+0.3)/(0.3*pre+rec),1)
    #with open(snap_root+'result.txt','a+') as f:
    #    f.write(str(F)+'\n'+str(mae)+'\n')
    with open(output_name,'wb') as f:
        pickle.dump({'pre':pre,'rec':rec,'mae':mae,'F':F,'time1':time_1,'time2':time_2},f,pickle.HIGHEST_PROTOCOL)
    #plt.figure()
    #plt.plot(rec,pre)
    #plt.savefig(snap_root+'result.png')
#print('F:',np.max(pre*rec*(1+0.3)/(0.3*pre+rec),1))
'''
'''
#output the conv3_i and conv3_dsni to see

for i in range(2000):
    net.forward()
    plt.figure()
    for iter,j in enumerate([1,3,5,7,9,11]):
        plt.subplot(4,3,j)
        if j in [1,3]:
            tem = net.blobs[f'conv{iter+1}_2'].data[0][0]
            tem = 1/(1+np.exp(-tem))
        elif j ==11:
            tem = net.blobs['pool5a'].data[0][0]
            tem = 1/(1+np.exp(-tem))
        else:
            tem = net.blobs[f'conv{iter+1}_3'].data[0][0]
            tem = 1/(1+np.exp(-tem))
        plt.imshow(tem,cmap = cm.Greys_r)
        plt.subplot(4,3,j+1)
        tem = net.blobs[f'conv3-dsn{iter+1}'].data[0][0]
        tem = 1/(1+np.exp(-tem))
        plt.imshow(tem,cmap =cm.Greys_r)
    assert i==1
    plt.savefig(f'./each_layer_output/{i}.png')
'''

#get the result for test_data
net = caffe.Net(code_root+prototxt_name,snap_root+caffemodel_name,caffe.TEST)
for j in range(1000):
    print(j)
    net.forward()
    plt.figure()
    #get sigmod output
    fin = net.blobs['sigmoid-fuse'].data[0][0].copy()
    plt.subplot(3,3,1)
    plt.imshow(fin*255,cmap=cm.Greys_r)
    for i in [2,3,4]:
        fin += net.blobs[f'sigmoid-dsn{i}'].data[0][0]
    fin /=4
    fin *= 255
    plt.subplot(3,3,2)
    plt.imshow(fin,cmap=cm.Greys_r)
    for i in range(1,7):
        plt.subplot(3,3,i+2)
        plt.imshow(net.blobs[f'sigmoid-dsn{i}'].data[0][0]*255,cmap = cm.Greys_r)
    plt.subplot(3,3,9)
    plt.imshow(net.blobs['label'].data[0][0]*255,cmap=cm.Greys_r)
    if(not os.path.exists(model_name)):
        os.mkdir(model_name)
    plt.savefig(model_name+f'/{j}.png')
    plt.close()
    plt.figure()
    neg = []
    for i in range(1,5):
        if(net.params[f'sal-conv4-dsn{i}'][0].data[0,0,0,0]<0):
            neg.append(i)
    for i in range(1,7):
        plt.subplot(2,3,i)
        out = net.blobs[f'conv3-dsn{i}'].data[0][0].copy()
        out = 1/(1+np.exp(-out))
        if i in neg:
            out = 1-out
        out *= 255
        plt.imshow(out,cmap=cm.Greys_r)
    #plt.savefig(model_name+f'/{j}_.png')
    #plt.close()
    plt.show()

