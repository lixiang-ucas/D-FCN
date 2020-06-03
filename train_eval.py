#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !nvidia-smi


# In[2]:
      


# In[3]:


import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow as tf
import socket
import importlib
import matplotlib.pylab as plt
import os

os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="2"

import sys
BASE_DIR = os.path.abspath('')
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import provider
import tf_util


# In[4]:


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=7, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='DFCN_pointnet2_group2', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_wen_v16_sample8192_group2_lw14_F1_noheight', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=2000, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=20000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args(args=[])


# In[5]:


EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp models/DFCN_pointnet2.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp tf_utils/DFCN_util_xy2.py %s' % (LOG_DIR)) # bkp of model def
# os.system('cp train_pointsift_lx_npsplit_V16.ipynb %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 9


# In[6]:
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def Acc_from_confusions(confusions):
    
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)
    
    mAcc = np.sum(TP)/np.sum(confusions)
    
    precision = TP / (TP_plus_FP + 1e-6)
    recall = TP / (TP_plus_FN+ 1e-6)
    fscore = 2*(precision * recall)/(precision + recall + 1e-6)
    
    ave_F1 = np.mean(fscore)
    
    s = 'Overall accuracy：{:5.2f}  Average F1 score：{:5.2f} \n'.format(100 * mAcc, 100 * ave_F1)
    s += log_acc(precision)
    s += log_acc(recall)
    s += log_acc(fscore)
    
    log_string(s)
    
    return mAcc, ave_F1
    

def log_acc(acc_list):
    s = ""
    for acc in acc_list:
        s += '{:5.2f} '.format(100 * acc)
    s += '\n'
    return s

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
    
def drawPlot(x,y,name):
    plt.rcParams['savefig.dpi'] = 300 
    plt.plot(np.arange(0,len(x)),x,'k-',alpha=1,label='Train max: '+str(round(max(x),3))+', min: '+str(round(min(x),3)))
    plt.plot(np.arange(0,len(y)),y,'r-',alpha=1,label='Test max: '+str(round(max(y),3))+', min: '+str(round(min(y),3)))
    plt.legend()
    plt.xlabel('epoch',fontsize=9)
    plt.ylabel(name+' value',fontsize=9)
    plt.savefig(LOG_DIR+"/"+name+".png",bbox_inches='tight')
    plt.show()
    
def drawF1Plot(x,name):
    plt.rcParams['savefig.dpi'] = 300 
    plt.plot(np.arange(0,len(x)),x,'k-',alpha=1,label='Train max: '+str(round(max(x),3))+', min: '+str(round(min(x),3)))
    plt.legend()
    plt.xlabel('epoch',fontsize=9)
    plt.ylabel(name+' value',fontsize=9)
    plt.savefig(LOG_DIR+"/"+name+".png",bbox_inches='tight')
    plt.show()
    
    
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, None, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(None, None))
    smpws_pl = tf.placeholder(tf.float32, shape=(None, None))
    return pointclouds_pl, labels_pl, smpws_pl

def pc_normalize_min_max(data):
    mindata = np.min(data[:,:3], axis=0)
    maxdata = np.max(data[:,:3], axis=0)
    return 2*(data[:,:3] - mindata)/(maxdata - mindata)

def pc_normalize_min(data):
    mindata = np.min(data[:,:3], axis=0)
    
    return (data[:,:3] - mindata)

def get_batch(dataset, index, npoints = NUM_POINT):
  
    if(dataset =='train'):
        cub_l = 30.0
        cub_w = 30.0
        cub_h = 100.0
        point_set =  trainSet[:,:3] - np.min(trainSet[:,:3], axis=0)
        semantic_seg = trainSet[:,4].astype(np.int32)
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        smpmin = np.maximum(coordmax-[cub_l,cub_w,cub_h], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[cub_l,cub_w,cub_h])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[cub_l/2,cub_w/2,cub_h/2]
            curmax = curcenter+[cub_l/2,cub_w/2,cub_h/2]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-0.0))*(point_set<(curmax+0.0)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            cur_feat_set = trainFeats[curchoice,:]
    #         if len(cur_semantic_seg)<npoints:
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set>=(curmin-0.0))*(cur_point_set<(curmax+0.0)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:2]-curmin[:2])/(curmax[:2]-curmin[:2])*[31.0,31.0])
            vidx = np.unique(vidx[:,0]*31.0+vidx[:,1])
            isvalid = np.sum(cur_semantic_seg>-1)/1.0/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0>=0.3
#             print('isvalid', isvalid,len(vidx)/31.0/31.0,np.sum(cur_semantic_seg>-1),len(cur_semantic_seg))
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
        point_set = cur_point_set[choice,:]
        feature_set = cur_feat_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = labelweights[semantic_seg]
        sample_weight *= mask
        return point_set, semantic_seg, sample_weight,feature_set
    
    if(dataset =='test'):

        cur_point_set = test_xyz[index]
        cur_semantic_seg = test_label[index].astype(np.int32)
        feature_set = test_feats[index]

        point_set = pc_normalize_min(cur_point_set)
        semantic_seg = cur_semantic_seg # N
        sample_weight = labelweights_t[semantic_seg]
    
        point_sets = np.expand_dims(point_set,0) # 1xNx3
        feature_set = np.expand_dims(feature_set,0) # 1xNx3
        semantic_segs = np.expand_dims(semantic_seg,0)  # 1xN
        sample_weights = np.expand_dims(sample_weight,0)  # 1xN
        return point_sets, semantic_segs, sample_weights,feature_set
    
def get_batch_wdp(dataset, batch_idx):
    bsize = BATCH_SIZE
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_feats = np.zeros((bsize, NUM_POINT, 1))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw,feat = get_batch('train',index=0)
        ps = pc_normalize_min(ps)
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
        batch_feats[i,:] = feat

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
        
    return batch_data, batch_label, batch_smpw, batch_feats   


# In[7]:


import pickle as pickle
import numpy as np

# train_f = open('./Data/train_merge_min_norm_fea.pickle', 'rb')
train_xyz, train_label, train_feats = pickle.load(train_f, encoding='bytes')
# train_xyz, train_label, train_feats = pickle.load(train_f)
train_f.close()

test_f = open('./Data/test_merge_min_norm_fea_paper_height.pickle', 'rb')
test_xyz, test_label, test_feats = pickle.load(test_f, encoding='bytes')
test_feats = [tt[:,1:2] for tt in test_feats] #reflectance
test_f.close()

NUM_CLASSES = 9
label_values = range(NUM_CLASSES)

trainSet = np.loadtxt('./Data/train_height.pts',skiprows=1)

label_w = trainSet[:,4].astype('uint8')
trainSet[:,3] = trainSet[:,3]/trainSet[:,3].max() #height above ground
trainSet[:,5] = trainSet[:,5]/trainSet[:,5].max() #reflectance

# trainFeats = trainSet[:,[3,5]] #use reflectance and height above ground
trainFeats = trainSet[:,5:6] #only use reflectance

labelweights = np.zeros(9)
tmp,_ = np.histogram(label_w,range(10))
labelweights = tmp
labelweights = labelweights.astype(np.float32)
labelweights = labelweights/np.sum(labelweights)
labelweights = 1/np.log(1.4+labelweights)
print(labelweights)

labelweights_t = np.ones(9)
print(labelweights_t)


# In[11]:


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    
   # Shuffle train samples
    train_idxs = np.arange(0, len(train_xyz))
    np.random.shuffle(train_idxs)
    num_batches = len(train_xyz)//BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        batch_data, batch_label, batch_smpw, batch_feats = get_batch_wdp('train', batch_idx)
        
        if batch_idx % (num_batches/2) == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        
        aug_data = provider.rotate_point_cloud_z(batch_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['feature_pl']: batch_feats,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, lr_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['learnrate']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        
    log_string('learn rate: %f' % (lr_val))
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    
    mloss = loss_sum / float(num_batches)
    macc = total_correct / float(total_seen)
    return mloss, macc

def eval_one_epoch_whole_scene(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    
    test_idxs = np.arange(0, len(test_xyz))
    
    TEST_BATCH_SIZE = 1
    num_batches = len(test_xyz)
    
    Confs = []
    
    
    is_continue_batch = False
    
    for batch_idx in range(num_batches):
        
        batch_data, batch_label, batch_smpw, batch_feats = get_batch('test', batch_idx)
        
#         print('Current start end /total batch num: %d %d/%d'%(start_idx, end_idx, num_batches))
        
        aug_data = batch_data
        
#         aug_data = provider.rotate_point_cloud_z(batch_data)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['feature_pl']: batch_feats,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, lr_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred'], ops['learnrate']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += batch_data.shape[1]
        loss_sum += loss_val
        
        NUM_POINT_fact = batch_data.shape[1]
        for i in range(TEST_BATCH_SIZE):
            for j in range(NUM_POINT_fact):
                l = batch_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i, j] == l)
                
        from sklearn.metrics import confusion_matrix
        Confs += [confusion_matrix(batch_label.flatten(), pred_val.flatten(), label_values)]
        
    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
    oa, avgF1 = Acc_from_confusions(C)
    
    log_string('learn rate: %f' % (lr_val))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    mloss = loss_sum / float(num_batches)
    macc = total_correct / float(total_seen)
    return oa, avgF1


# In[12]:
pointclouds_pl, labels_pl, smpws_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
feature_pl = tf.placeholder(tf.float32, shape=(None, None, 1))
is_training_pl = tf.placeholder(tf.bool, shape=())

# Note the global_step=batch parameter to minimize. 
# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
batch = tf.Variable(0)
bn_decay = get_bn_decay(batch)
tf.summary.scalar('bn_decay', bn_decay)

# Get model and loss 
pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay, feature=feature_pl)
loss = MODEL.get_loss(pred, labels_pl, smpws_pl)

tf.summary.scalar('loss', loss)

correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
tf.summary.scalar('accuracy', accuracy)

# Get training operator
learning_rate = get_learning_rate(batch)
tf.summary.scalar('learning_rate', learning_rate)
if OPTIMIZER == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
elif OPTIMIZER == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=batch)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
sess = tf.Session(config=config)

# Add summary writers
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                          sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

# Init variables
init = tf.global_variables_initializer()
sess.run(init, {is_training_pl:True})

ops = {'pointclouds_pl': pointclouds_pl,
       'labels_pl': labels_pl,
       'feature_pl': feature_pl,
       'smpws_pl': smpws_pl,
       'is_training_pl': is_training_pl,
       'pred': pred,
       'loss': loss,
       'train_op': train_op,
       'merged': merged,
       'step': batch,
       'learnrate': learning_rate}

train_acc_list=[]
test_acc_list=[]
train_loss_list=[]
test_F1_list=[]


# In[ ]:


best_F1 = -1

for epoch in range(MAX_EPOCH):
    log_string('**** EPOCH %03d ****' % (epoch))
    sys.stdout.flush()

    train_loss, train_acc = train_one_epoch(sess, ops, train_writer)
    test_acc, test_F1 = eval_one_epoch_whole_scene(sess, ops, test_writer)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    test_F1_list.append(test_F1)

    
    drawF1Plot(test_F1_list, "F1_score")
    
    drawPlot(train_acc_list,test_acc_list,"Accuracy")

    # Save the variables to disk.
    
    if test_F1 > best_F1:
        best_F1 = test_F1
        save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
        log_string("Model saved in file: %s" % save_path)
                
    if epoch % 10 == 0:
        save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
        log_string("Model saved in file: %s" % save_path)


# In[ ]:


# checkpoint_path = 'log_lx_nosplit_v16/best_model_epoch_1107.ckpt'
# saver.restore(sess, checkpoint_path)


# # In[ ]:


# test_loss, test_acc = eval_one_epoch_whole_scene(sess, ops, test_writer)

