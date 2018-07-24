import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
caffe_root = '../caffe_dss-master/'
code_root  = caffe_root+'examples/dss_final/'
sys.path.insert(0,caffe_root+'python')
import caffe
import pydensecrf.densecrf as dcrf
import os 
import pickle
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

def evaluate(net,ID,num,output_dir):
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]=ID
    #caffe.set_mode_gpu()
    #iters = x*interval+interval
    #net = caffe.Net(proto,'./result/tmp.caffemodel',caffe.TEST)
    #net = caffe.Net(code_root+prototxt_name,snap_root+,caffe.TEST)
    net.copy_from(output_dir+'tmp.caffemodel')
    output_layer = [k for k in net.blobs.keys() if 'sigmoid' in k]
    out_num = len(output_layer)
    pre = np.zeros([out_num,256])
    rec = np.zeros([out_num,256])
    mae = np.zeros(out_num)
    time1 = []
    time2 = []
    for i in range(num):
        #print('num:',i)
        #time_1 = time.time()
        net.forward()
        #print(i)
        #time1.append(time.time()-time_1)
        out = []
        for l in output_layer:
            out.append(net.blobs[l].data[0][0])
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
        #img = untransform(net.blobs['data'].data[0])
        for j,fin2 in enumerate(out):
            mae[j] += compute_mae(fin2,label)
            fin4 = fin2 *255
            compute_pr(fin4,net.blobs['label'].data[0][0],pre[j,:],rec[j,:])
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
    maxF = np.max(pre*rec*(1+0.3)/(0.3*pre+rec),1)
    meanF = np.mean(pre*rec*(1+0.3)/(0.3*pre+rec),1)
    #with open(snap_root+'result.txt','a+') as f:
    #    f.write(str(F)+'\n'+str(mae)+'\n')
    #with open(output_name,'wb') as f:
    #    pickle.dump({'pre':pre,'rec':rec,'mae':mae,'F':F,'time1':time_1,'time2':time_2},f,pickle.HIGHEST_PROTOCOL)
    return maxF,meanF,mae

def get_image(net,output_dir,num,neg,display_layer):
    sigmoid_output = [k for k in net.blobs.keys() if 'sigmoid' in k]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for j in range(num):
        print(j)
        #assert j==0
        #get the sigmoid output
        sigmoid_output = [k for k in net.blobs.keys() if 'sigmoid' in k]
        net.forward()
        plt.figure()
        for n,i in enumerate(sigmoid_output):
            plt.subplot(4,4,n+1)
            tem = net.blobs[i].data[0][0]
            plt.imshow(tem,cmap = cm.Greys_r)
        image = net.blobs['data'].data[0]
        image = untransform(image)
        label = net.blobs['label'].data[0][0]*255
        plt.subplot(4,4,15)
        plt.imshow(image)
        plt.subplot(4,4,16)
        plt.imshow(label,cmap = cm.Greys_r)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)
        plt.savefig(output_dir+f'/{j}.png')
        #plt.show()
        plt.close()
        
        plt.figure()
        for i,layer in enumerate(display_layer):
            plt.subplot(2,3,i+1)
            out = net.blobs[layer].data[0][0].copy()
            out = out*neg[i]
            out = 1/(1+np.exp(-out))
            out *= 255
            plt.imshow(out,cmap=cm.Greys_r)
        plt.savefig(output_dir+f'/{j}_.png')
        #plt.show()
        plt.close()
        

def test(net,num,output_dir,crf):
    output_layer = [k for k in net.blobs.keys() if 'sigmoid' in k]
    out_num = len(output_layer)
    pre = np.zeros([out_num,256])
    rec = np.zeros([out_num,256])
    mae = np.zeros(out_num)
    time1 = []
    time2 = []
    for i in range(num):
        net.forward()
        out = []
        print('test:',i)
        for l in output_layer:
            out.append(net.blobs[l].data[0][0])
        #fin = net.blobs['sigmoid-fuse'].data[0][0].copy()
        #for m in range(3):
        #    fin += net.blobs[f'sigmoid-dsn{m+2}'].data[0][0]
        #fin = fin/4
        #out.append(fin)
        image = net.blobs['data'].data[0]
        image  = untransform(image)
        label = net.blobs['label'].data[0][0]
        #mae += compute_mae(fin,label)
        #print(mae)
        #img = untransform(net.blobs['data'].data[0])
        for j,fin2 in enumerate(out):
            if crf:
                fin2 = crf_compute(image,fin2)
            mae[j] += compute_mae(fin2,label)
            fin4 = fin2 *255
            compute_pr(fin4,net.blobs['label'].data[0][0],pre[j,:],rec[j,:])
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
    maxF = np.max(pre*rec*(1+0.3)/(0.3*pre+rec),1)
    meanF = np.mean(pre*rec*(1+0.3)/(0.3*pre+rec),1)
    
    with open(output_dir+'result.txt','a+') as f:
        f.write(str(maxF)+'\n'+str(mae)+'\n'+str(meanF)+'\n')
    with open(output_dir+'resut.pickle','wb') as f:
        pickle.dump({'pre':pre,'rec':rec,'mae':mae,'maxF':maxF,'meanF':meanF},f,pickle.HIGHEST_PROTOCOL)
    print(maxF,'\n',meanF,'\n',mae)


