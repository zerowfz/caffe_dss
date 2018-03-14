import sys
import numpy as np
caffe_root = '../caffe_dss-master/'
code_root  = caffe_root+'examples/dss_final/'
sys.path.insert(0,caffe_root+'python')
import caffe
import os 
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


def evaluate(net,ID,num,output_dir):
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=ID
    #caffe.set_mode_gpu()
    #iters = x*interval+interval
    net.copy_from(output_dir+'tmp.caffemodel')
    #net = caffe.Net(proto,'./result/tmp.caffemodel',caffe.TEST)
    #net = caffe.Net(code_root+prototxt_name,snap_root+,caffe.TEST)
    pre = np.zeros(256)
    rec = np.zeros(256)
    mae = 0
    time1 = []
    time2 = []
    for i in range(num):
        #print('num:',i)
        #time_1 = time.time()
        net.forward()
        #time1.append(time.time()-time_1)
        out = []
        out.append(net.blobs['sigmoid-fuse'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn6'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn5'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn4'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn3'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn2'].data[0][0])
        #out.append(net.blobs['sigmoid-dsn1'].data[0][0])
        #fin = net.blobs['sigmoid-fuse'].data[0][0].copy()
        #for m in range(3):
        #    fin += net.blobs[f'sigmoid-dsn{m+2}'].data[0][0]
        #fin = fin/4
        #out.append(fin)
        label = net.blobs['label'].data[0][0]
        #mae += compute_mae(fin,label)
        #print(mae)
        img = untransform(net.blobs['data'].data[0])
        for j,fin2 in enumerate(out):
            mae += compute_mae(fin2,label)
            fin4 = fin2 *255
            compute_pr(fin4,net.blobs['label'].data[0][0],pre,rec)
            #time_2 = time.time()
            #fin3 = crf_compute(img,fin2)
            #time2.append(time.time()-time_2)
            #mae += compute_mae(fin3,label)
            #fin3 =fin3 * 255
            #compute_pr(fin3,net.blobs['label'].data[0][0],pre[j+1,:],rec[j+1,:])
            #print('xxxxx:',j,'xxxx:',pre[j,255])
    pre = pre/num
    rec = rec/num
    mae = mae/num
    #time_1 = np.array(time1).mean()
    #time_2 = np.array(time2).mean()
    maxF = np.max(pre*rec*(1+0.3)/(0.3*pre+rec))
    meanF = np.mean(pre*rec*(1+0.3)/(0.3*pre+rec))
    #with open(snap_root+'result.txt','a+') as f:
    #    f.write(str(F)+'\n'+str(mae)+'\n')
    #with open(output_name,'wb') as f:
    #    pickle.dump({'pre':pre,'rec':rec,'mae':mae,'F':F,'time1':time_1,'time2':time_2},f,pickle.HIGHEST_PROTOCOL)
    return maxF,meanF,mae

