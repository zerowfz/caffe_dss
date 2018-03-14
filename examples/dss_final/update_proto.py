import sys
from google.protobuf import text_format
from collections import OrderedDict,Counter
caffe_root = '../../'
sys.path.insert(0,caffe_root+'python')
from caffe import layers as L,params as P,to_proto
from caffe.proto import caffe_pb2
from caffe import NetSpec
from caffe.net_spec import assign_proto

model = caffe_pb2.NetParameter()
f = open('multi-2.prototxt','r')
text_format.Merge(str(f.read()),model)
for i in model.layer:
    if '-conv3' in i.name:
        i.param[0].lr_mult = 0.01
        i.param[1].lr_mult = 0.02

with open('multi-2-0.01.prototxt','w') as f:
    f.write(str(model))

'''
j=0
for i in model.layer:
    if (i.name=='pool4'):
        j=1
    if i.name=='sal-loss-dsn5':
        j=0
    if j==1:
        i.name = 'sal-'+i.name
with open('trainnet2.prototxt','w') as f:
    f.write(str(model))
'''     

