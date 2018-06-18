from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, fbeta_score, confusion_matrix, precision_recall_fscore_support, confusion_matrix
from skmultilearn.problem_transform import LabelPowerset # to help stratify while sampling
import numpy as np
import tensorflow as tf
import glob
import time
import os
import itertools
import matplotlib.pyplot as plt
from tensorflow.python.training import queue_runner
import csv
import scipy.io as sio

# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input], shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNExperiment(object):
    def __init__(self, dataset_name, model_name, net_constructor):
        # Initialize all defaults
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_iterations = 450000
        self.iterations_per_test = 525 #train size = 1050, half of it
        self.display_iter = 50
        self.snapshot_iter = 500
        self.train_batch_size = 50
        self.val_batch_size = 25
        self.test_batch_size = 0.2*2100
        self.crop_if_possible = True
        self.debug = False
        self.starter_learning_rate = 0.01
        self.learning_rate_exp = 0.97
        self.learning_rate_step = 1000
        self.reports = {}
        self.silent = False
        self.optimizer = 'momentum'
        self.kFold = False
        self.extract = False #to extract features from all data points
        if self.extract==True:
            self.train_batch_size = 2100
        
        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        self.net.extract = self.extract
        self.net_desc = GraphCNNNetworkDescription()
        tf.reset_default_graph()
        self.config = tf.ConfigProto()
        #self.config.gpu_options.allocator_type = 'BFC'
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.2


        
    # print_ext can be disabled through the silent flag
    def print_ext(self, *args):
        if self.silent == False:
            print_ext(*args)
            
    # Will retrieve the value stored as the maximum test accuracy on a trained network
    # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
    def get_max_accuracy(self):
        tf.reset_default_graph()
        with tf.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it
    
        # Run all folds in a CV and calculate mean/std
    def run_experiments(self, beta=1.0, test_split=0.2, threshold=0.5):
        self.net_constructor.create_network(self.net_desc, []) #calling th function in run_graph file to create network
        desc = self.net_desc.get_description()
        self.print_ext('Running CV for:', desc)
        start_time = time.time()
        self.limit = 0.5 - threshold
        tf.reset_default_graph() #check this line
        self.create_test_train(test_split=test_split)
        f_score = self.run(beta=beta)
        self.print_ext('fscore is: %f' % (f_score))
        acc = f_score
        
        verify_dir_exists('./results/')
        with open('./results/%s.txt' % self.dataset_name, 'a+') as file:
            file.write('%s\t%s\t%d seconds\t%.2f\n' % (str(datetime.now()), desc, time.time()-start_time, acc))
        return acc
        
    # Prepares samples for experiment, accepts a list (vertices, adjacency, labels) where:
    # vertices = list of NxC matrices where C is the same over all samples, N can be different between samples
    # adjacency = list of NxLxN tensors containing L NxN adjacency matrices of the given samples
    # labels = list of sample labels
    # len(vertices) == len(adjacency) == len(labels)
    def preprocess_data(self, dataset):
        features = np.squeeze(dataset[0])
        edges = np.squeeze(dataset[1])
        labels = np.squeeze(dataset[2])
        self.weights = np.squeeze(dataset[3])
        self.label_wt = np.squeeze(dataset[4])
        if self.extract == True:
            index = np.squeeze(dataset[5])
            classes = np.squeeze(dataset[6])
            
        self.graph_size = np.array([s.shape[0] for s in edges]).astype(np.int64)
        
        self.largest_graph = max(self.graph_size)
        self.smallest_graph = min(self.graph_size)
        print_ext('Largest graph size is %d and smallest graph size is %d' % (self.largest_graph,self.smallest_graph))
        self.print_ext('Padding samples')
        self.graph_vertices = np.zeros((len(dataset[0]), self.largest_graph, np.shape(features[0])[1]))
        self.graph_adjacency = np.zeros((len(dataset[0]), self.largest_graph, self.largest_graph))
        self.index = np.zeros((len(dataset[0])))
        self.classes = np.zeros((len(dataset[0])))
        for i in range(len(dataset[0])):
        # for i in range(2):
            # pad all vertices to match size
            self.graph_vertices[i,:,:] = np.pad(features[i].astype(np.float32), ((0,self.largest_graph-dataset[0][i].shape[0]), (0, 0)), 'constant', constant_values=(0))

            # pad all adjacency matrices to match size
            self.graph_adjacency[i,:,:] = np.pad(edges[i].astype(np.float32), ((0, self.largest_graph-dataset[1][i].shape[0]), (0, self.largest_graph-dataset[1][i].shape[1])), 'constant', constant_values=(0))
            
            if self.extract == True:
                # removing the extra dimension from every element
                self.index[i] = np.squeeze(index[i])
                self.classes[i] = np.squeeze(classes[i])

        self.graph_adjacency = np.expand_dims(self.graph_adjacency,axis=2)
        self.graph_adjacency = np.array(self.graph_adjacency, dtype='float32')
        self.graph_vertices = np.array(self.graph_vertices, dtype='float32')
        # self.print_ext("Shape of graph_vertices is:",np.shape(self.graph_adjacency[np.ones(15,dtype=int),:,:]))
        self.print_ext('Stacking samples')
             self.graph_labels = labels.astype(np.int64)
        # self.ind = 0:2099;
        
        self.no_samples = self.graph_labels.shape[0]
        
        single_sample = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
        

        
        
    def create_test_train(self, test_split=0.2):
      
        indC = range(0,2100)
        
        classnum = range(1,22)
        classNum = np.tile(classnum,[100,1])
        classNum = np.transpose(classNum,(1,0))
        classNum = classNum.flatten()

        # test-train split (with stratify)
        rem_idx, self.test_idx = train_test_split(indC, test_size=test_split, random_state=120, stratify=classNum) #50-30-20 split
        # train_idx_leave, test_idx_leave = train_test_split(indx_mat[2], test_size=test_split, random_state=120)
        # self.test_idx = np.array(np.ma.append(test_idx,test_idx_leave))
        rem_idx = np.array(rem_idx, dtype = np.int32)
        
        indC_rem = [indC[i] for i in rem_idx]
        classNum_rem = classNum[rem_idx]
        
                                     
        # train-val split
        self.train_idx, self.val_idx = train_test_split(indC_rem, test_size=0.375, random_state=120, stratify=classNum_rem)
        
        # self.train_idx = np.array(np.ma.append(train_idx,train_idx_leave))
        # self.val_idx = np.array(np.ma.append(val_idx,val_idx_leave))
        
        self.train_idx = np.array(self.train_idx, dtype = int)
        self.val_idx = np.array(self.val_idx, dtype = int)
        self.test_idx = np.array(self.test_idx, dtype=int)
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_val = self.val_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_val:', self.no_samples_val, 'no_samples_test:', self.no_samples_test)
        
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.val_batch_size == 0:
            self.val_batch_size = self.no_samples_val
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        self.train_batch_size = min(self.train_batch_size, self.no_samples_train)
        self.val_batch_size = np.array(min(self.val_batch_size, self.no_samples_val), dtype=int)
        self.test_batch_size = np.array(min(self.test_batch_size, self.no_samples_test), dtype=int)
        sio.savemat('test_train_idx.mat',{'train_ind' : self.train_idx, 'val_ind': self.val_idx, 'test_ind': self.test_idx})
        
    # This function is cropped before batch
    # Slice each sample to improve performance
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1]), tf.int64))
        vertices.set_shape([None, self.graph_vertices.shape[2]])
        adjacency = tf.slice(single_sample[1], np.array([0, 0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1, single_sample[3]]), tf.int64))
        adjacency.set_shape([None, self.graph_adjacency.shape[2], None])
        
        # V, A, labels, mask
        return [vertices, adjacency, single_sample[2], tf.expand_dims(tf.ones(tf.slice(tf.shape(vertices), [0], [1])), axis=-1)]
        
    def create_input_variable(self, input):
        for i in range(len(input)):
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        return input

        
    # Create input_producers and batch queues
    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    training_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    self.print_ext('Shape of graph_vertices is:',np.shape(self.graph_vertices))
                    self.print_ext('Shape of graph_adjacency is:',np.shape(self.graph_adjacency))
                    self.print_ext('Shape of train_idx is:',np.shape(self.train_idx))
                    
                    #expanding dimension of weights to help broadcast
                    self.weights = np.expand_dims(self.weights,axis=-1)
                    self.weights = np.expand_dims(self.weights,axis=-1)
                    
                    # if self.kFold:
                    # add self.freq_classes
                    if self.extract == True:
                        vertices = self.graph_vertices
                        adjacency = self.graph_adjacency
                        labels = self.graph_labels
                        weights = self.weights
                        graph_size = self.graph_size
                        self.print_ext('saving features mode')
                    else:
                        vertices = self.graph_vertices[self.train_idx,:,:]
                        adjacency = self.graph_adjacency[self.train_idx, :, :, :]
                        labels = self.graph_labels[self.train_idx, :]
                        weights = self.weights[self.train_idx,:,:]
                        graph_size = self.graph_size[self.train_idx]
                    # else:
                    #     vertices = self.graph_vertices
                    #     adjacency = self.graph_adjacency
                    #     # adjacency = adjacency[:, :, :, self.train_idx]
                    #     labels = self.graph_labels
                    #     graph_size = self.graph_size
                    # input_mask = np.zeros([1, self.largest_graph, 1]).astype(np.float32)
                    # input_mask[:, self.train_idx, :] = 1
                    # training_samples = [s[self.train_idx] for s in training_samples]
                    if self.extract == True:
                        training_samples = [vertices, adjacency, labels, weights, self.index, self.classes]
                    else:
                        training_samples = [vertices, adjacency, labels, weights]
                    
                    if self.crop_if_possible == False:
                        training_samples[3] = get_node_mask(training_samples[3], max_size=self.graph_vertices.shape[1])
                        
                    # Create tf.constants
                    training_samples = self.create_input_variable(training_samples)
                    
                    # Slice first dimension to obtain samples
                    if self.extract == True:
                        single_sample=tf.train.slice_input_producer(training_samples,shuffle=False,num_epochs=1,capacity=self.train_batch_size)
                        # single_sample=tf.train.slice_input_producer(training_samples,shuffle=False,capacity=self.train_batch_size)
                        self.print_ext('saving features mode')
                    else:
                        single_sample=tf.train.slice_input_producer(training_samples,shuffle=True,capacity=self.train_batch_size)
                    
                    # Cropping samples improves performance but is not required
                    # if self.crop_if_possible:
                        # self.print_ext('Cropping smaller graphs')
                        # single_sample = self.crop_single_sample(single_sample)
                    
                    # creates training batch queue
                    if self.extract == True:
                        train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=1)
                    else:
                        train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=1)

                # Create the val queue
                with tf.variable_scope('val_data') as scope:
                    self.print_ext('Creating val Tensorflow Tensors')
                    
                    # Create tensor with all test samples
                    vertices = self.graph_vertices[self.val_idx, :, :]
                    adjacency = self.graph_adjacency[self.val_idx, :, :, :]
                    # adjacency = adjacency[:, :, :, self.train_idx]
                    labels = self.graph_labels[self.val_idx, :]
                    weights = self.weights[self.val_idx,:,:]
                    graph_size = self.graph_size[self.val_idx]
                    index = self.index[self.val_idx]
                    classes = self.classes[self.val_idx]
    
                    
                    val_samples = [vertices, adjacency, labels, weights]
                    # If using mini-batch we will need a queue 
                    # if self.val_batch_size != self.no_samples_val:
                    if 1:
                        if self.crop_if_possible == False:
                            val_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        val_samples = self.create_input_variable(val_samples)
                        
                        single_sample = tf.train.slice_input_producer(val_samples, shuffle=True, capacity=self.val_batch_size)
                        # if self.crop_if_possible:
                            # single_sample = self.crop_single_sample(single_sample)
                            
                        val_queue = _make_batch_queue(single_sample, capacity=self.val_batch_size*2, num_threads=1)
                        
                    # If using full-batch no need for queues
                    else:
                        val_samples[3] = get_node_mask(val_samples[3], max_size=self.graph_vertices.shape[1])
                        
                        val_samples = self.create_input_variable(val_samples)
                        for i in range(len(val_samples)):
                            var = tf.cast(val_samples[i],tf.float32)
                            val_samples[i] = var
                                     
                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all test samples
                    # test_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    # test_samples = [s[self.test_idx] for s in test_samples]
                    # if self.kFold:
                    vertices = self.graph_vertices[self.test_idx, :, :]
                    adjacency = self.graph_adjacency[self.test_idx, :, :, :]
                    # adjacency = adjacency[:, :, :, self.train_idx]
                    labels = self.graph_labels[self.test_idx, :]
                    weights = self.weights[self.test_idx,:,:]
                    graph_size = self.graph_size[self.test_idx]
                    index = self.index[self.test_idx]
                    classes = self.classes[self.test_idx]
 
                    
                    test_samples = [vertices, adjacency, labels, weights]
                        
                    # If using mini-batch we will need a queue 
                    # if self.test_batch_size != self.no_samples_test:
                    if 1:
                        if self.crop_if_possible == False:
                            test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = self.create_input_variable(test_samples)
                        
                        single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size)
                        # if self.crop_if_possible:
                            # single_sample = self.crop_single_sample(single_sample)
                            
                        test_queue = _make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                        
                    # If using full-batch no need for queues
                    else:
                        test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        
                        test_samples = self.create_input_variable(test_samples)
                        for i in range(len(test_samples)):
                            var = tf.cast(test_samples[i],tf.float32)
                            test_samples[i] = var

                            # test_samples = tf.cast(test_samples, tf.int64)
                            
                # obtain batch depending on is_training and if test is a queue
                
                #if self.val_batch_size == self.no_samples_test:
                    #return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_samples)
                # self.net.is_training = 1 => train, 
                #                      = 0 => validate, 
                #                      = else => test
                if self.extract == True:
                    return tf.case(pred_fn_pairs=[
                         (tf.equal(self.net.is_training,1), lambda:train_queue.dequeue_many(self.train_batch_size))],  default=lambda:train_queue.dequeue_many(self.train_batch_size), exclusive=True)
                else:
                     return tf.case(pred_fn_pairs=[
                         (tf.equal(self.net.is_training,1), lambda:train_queue.dequeue_many(self.train_batch_size)),            (tf.equal(self.net.is_training,0),  lambda:val_queue.dequeue_many(self.val_batch_size)),
(tf.equal(self.net.is_training,-1), lambda:test_queue.dequeue_many(self.test_batch_size))],  default=lambda:train_queue.dequeue_many(self.train_batch_size), exclusive=True)
                

    # Function called with the output of the Graph-CNN model
    # Should add the loss to the 'losses' collection and add any summaries needed (e.g. accuracy) 
    def create_loss_function(self):
        with tf.variable_scope('loss') as scope:
            self.print_ext('Creating loss function and summaries')
            self.print_ext('Shape of logits is:',self.net.current_V.get_shape(),'and shape of labels is:',self.net.labels.get_shape())
            self.net.labels = tf.cast(self.net.labels,'float32')
            # epsilon = tf.constant(value=1e-10)
            # self.net.current_V = tf.add(self.net.current_V, epsilon)
            self.net.current_V_weighted = tf.multiply(self.net.current_V, self.label_wt)
            # casting label and prediction into float64 to remove tf reduce_mean error
            current_V_f64 = tf.cast(self.net.current_V,'float64')
            labels_f64 = tf.cast(self.net.labels,'float64')            
            
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.current_V, labels=self.net.labels))
            
            # tot_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.current_V, labels=self.net.labels)
            # mask = tf.cast(tf.squeeze(self.net.current_mask, axis=-1), dtype=tf.float32)
            # cross_entropy_weighted  = tf.reduce_mean(tf.matmul(tf.transpose(tot_loss,[1,0]), mask)) 
            
            cross_entropy_weighted  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.net.current_V_weighted, labels=self.net.labels))
            # cross_entropy_weighted  = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.net.current_V, targets=self.net.labels, pos_weight=mask)) 
            
            # cross_entropy = self.loss(logits=self.net.current_V, labels=self.net.labels)
            #making values beyond the threshold 1
            self.features = self.net.current_V
            self.net.current_V = tf.add(tf.nn.sigmoid(self.net.current_V), self.limit)
            self.net.current_V = tf.minimum(tf.round(self.net.current_V),1)
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            self.max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
            # max_acc = tf.cond(self.net.is_training, lambda: tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy)), lambda: tf.assign(self.max_acc_test, tf.maximum(self.max_acc_test, accuracy)))
            
            tf.add_to_collection('losses', cross_entropy)
            
            # tf.summary.scalar('accuracy', accuracy)
            # tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('cross_entropy_weighted', cross_entropy_weighted)
            # tf.summary.scalar('precision',prec)
            # tf.summary.scalar('recall',rec)
            # tf.summary.scalar('fscore',f_score)
            # tf.summary.scalar('hamm_score',hamm_score)                    
            tf.summary.histogram('prediction',self.net.current_V)
            tf.summary.histogram('actual_labels',self.net.labels)
            # tf.summary.histogram('difference',self.difference)
            
            # if silent == false display these statistics:
            # self.reports['accuracy'] = accuracy
            # self.reports['max acc.'] = max_acc
            self.reports['cross_entropy'] = cross_entropy
            self.reports['prediction'] = self.net.current_V
            self.reports['prediction_weighted'] = self.net.current_V_weighted
            self.reports['embed_features'] = self.net.embed_features
            self.reports['first_gc_features'] = self.net.first_gc_features
            self.reports['second_gc_features'] = self.net.second_gc_features
            self.reports['fc_features'] = self.net.fc_features
            self.reports['features'] = self.features
            self.reports['actual_labels'] = self.net.labels
            if self.extract == True:
                self.reports['index'] = self.net.index
                self.reports['classes'] = self.net.classes
            # self.reports['prec'] = prec
            # self.reports['rec'] = rec
            # self.reports['fscore'] = f_score
            # self.reports['hamm_score'] = hamm_score
            # self.reports['difference'] = self.difference
        
    # check if the model has a saved iteration and return the latest iteration step
    def check_model_iteration(self):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        return int(latest[len(self.snapshot_path + 'model-'):])
        
    # load_model if any checkpoint exist
    def load_model(self, sess, saver):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        print(latest)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % i)
        return i
        
    def save_model(self, sess, saver, i):
        if not os.path.exists(self.snapshot_path):
                os.makedirs(self.snapshot_path)
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        #if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
        if 1:
            self.print_ext('Saving model at %d' % i)
            #verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=i)
            self.print_ext('Model saved to %s' % result)
    
    
    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads
    def run(self, beta=1.0):
        self.variable_initialization = {}
        
        hamm_score_train = np.zeros((self.train_batch_size))
        hamm_score_test = np.zeros((self.val_batch_size))
        fbeta_train = np.zeros((self.train_batch_size))
        fbeta_test = np.zeros((self.val_batch_size))
        
        
        
        
        with open('./results/predictions_test.csv', 'w') as csvfile:
            fieldnames = ['actual_labels', 'predictions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        with open('./results/scores_test.csv', 'w') as csvfile:
            fieldnames = ['hamm_score', 'fbeta_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        with open('./results/predictions_train.csv', 'w') as csvfile:
            fieldnames = ['actual_labels', 'predictions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open('./results/scores_train.csv', 'w') as csvfile:
            fieldnames = ['hamm_score', 'fbeta_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
        path = '/src'
        
        self.print_ext('Training model "%s"!' % self.model_name)
        if self.kFold and hasattr(self, 'fold_id') and self.fold_id:
            self.snapshot_path = './snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
            self.test_summary_path = './summary/%s/test/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
            self.train_summary_path = './summary/%s/train/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
        else:
            # Directory name changed from snapshots to snapshots_test
            self.snapshot_path = path+'/snapshots/%s/%s/' % (self.dataset_name, self.model_name)
            self.test_summary_path = path+'/summary/%s/test/%s' %(self.dataset_name, self.model_name)
            self.train_summary_path = path+'/summary/%s/train/%s' %(self.dataset_name, self.model_name)
        
        if self.extract==False:
        # if 0:
            i = 0
        else:
            self.print_ext(self.snapshot_path)
            i = self.check_model_iteration()
        
        if i < self.num_iterations:
            self.print_ext('Creating training network')
            
            self.net.is_training = tf.placeholder(tf.int32, shape=())
            prec = tf.placeholder(tf.float32, shape=(), name='precision')
            rec = tf.placeholder(tf.float32, shape=(), name='recall')
            f_score = tf.placeholder(tf.float32, shape=(), name='f_score')
            hamm_loss = tf.placeholder(tf.float32, shape=(), name='hamm_loss')

            
            self.net.global_step = tf.Variable(i,name='global_step',trainable=False)
            


            tf.summary.scalar('precision',prec)
            tf.summary.scalar('recall',rec)
            tf.summary.scalar('fscore',f_score)
            tf.summary.scalar('hamm_loss',hamm_loss)
            
            
            input = self.create_data()
            self.net_constructor.create_network(self.net, input)
            self.create_loss_function()
            
            self.print_ext('Preparing training')
            loss = tf.add_n(tf.get_collection('losses'))
            if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            
            
            with tf.control_dependencies(update_ops):
                if self.optimizer == 'adam':
                    train_step = tf.train.AdamOptimizer(learning_rate=self.starter_learning_rate).minimize(loss, global_step=self.net.global_step)
                else:
                    self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.net.global_step, self.learning_rate_step, self.learning_rate_exp, staircase=True)
                    train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss, global_step=self.net.global_step)
                    self.reports['lr'] = self.learning_rate
                    tf.summary.scalar('learning_rate', self.learning_rate)
            
            
            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer(), self.variable_initialization)
                
                if 1:
                    saver = tf.train.Saver()
                    #self.load_model(sess, saver)
                    if self.kFold:
                        snapshot_path_latest = './snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % (self.fold_id-1))
                        latest = tf.train.latest_checkpoint(snapshot_path_latest)
                    
                    # Directory name changed from snapshots to snapshots_test
                    snapshot_path_latest2 = path+'/snapshots/%s/%s/' % (self.dataset_name, self.model_name)
                    latest2 = tf.train.latest_checkpoint(snapshot_path_latest2)
                    
                    if self.kFold and latest != None:
                        saver.restore(sess, latest)
                        cur_i = int(latest[len(snapshot_path_latest + 'model-'):])
                        print_ext('Restoring last models fold-%d checkpoint at %d' % ((self.fold_id-1),cur_i))
                    elif latest2 != None:
                        saver.restore(sess, latest2)
                        cur_i = int(latest2[len(snapshot_path_latest2 + 'model-'):])
                        print_ext('Restoring last models default checkpoint at %d' % cur_i)
                                
                    self.print_ext('Starting summaries')
                    if not os.path.exists(self.train_summary_path):
                        print_ext('Making dir for train summary')
                        os.makedirs(self.train_summary_path)
                    if not os.path.exists(self.test_summary_path):
                        os.makedirs(self.test_summary_path)
                    test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                    train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)
            
                summary_merged = tf.summary.merge_all()
            
                self.print_ext('Starting threads')
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'validation_batch_size:', self.val_batch_size)
                wasKeyboardInterrupt = False
                try:
                # if 1:
                    total_training = 0.0
                    total_testing = 0.0
                    start_at = time.time()
                    last_summary = time.time()
                    while i < self.num_iterations:
                        
                        #print_ext('%d out of %d done' % (i, self.num_iterations))
                        if i % self.snapshot_iter == 0:
                            print_ext('Saving iteration')
                            self.save_model(sess, saver, i)
                        if i % self.iterations_per_test == 0:
                            
                            start_temp = time.time()
                            reports = sess.run(self.reports, feed_dict={self.net.is_training:0})
                            # print_ext('calculating scores')
                            #calculating precision, recall, fscore and hamming score
                            hamm_score = hamming_loss(reports['actual_labels'],reports['prediction'])
                            # fbeta_val = fbeta_score(reports['actual_labels'],reports['prediction'], beta=beta,average='samples')
                            precision, recall, fscore, _ = precision_recall_fscore_support(reports['actual_labels'],reports['prediction'],average='samples')
                            # assign_op_prec = prec.assign(precision)
                            # assign_op_rec = rec.assign(recall)
                            # assign_op_fscore = f_score.assign(fscore)
                            # assign_op_hamm = hamm_loss.assign(hamm_score)
                            
                            # summary, _, _, _, _ = sess.run([summary_merged, assign_op_prec, assign_op_rec, assign_op_fscore, assign_op_hamm], feed_dict={self.net.is_training:0})
                            summary = sess.run(summary_merged, feed_dict={self.net.is_training:0, prec:precision, rec: recall, f_score:fscore, hamm_loss:hamm_score})
                            
                            # sio.savemat('gcn_features.mat', {'features':reports['features']})
                            total_testing += time.time() - start_temp
                            self.print_ext('Test Step %d Finished' % i)
                            
                            for key, value in reports.items():
                                if key != 'actual_labels' and key!='prediction' and key!='features' and key!='embed_features' and key!='prediction_weighted':
                                    self.print_ext('Test Step %d "%s" = ' % (i, key), value)
                                    
                            with open('./results/predictions_test.csv', 'a') as csvfile:
                                fieldnames = ['actual_labels', 'predictions']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'actual_labels': reports['actual_labels'], 'predictions': reports['prediction']})
                            with open('./results/scores_test.csv', 'a') as csvfile:
                                fieldnames = ['hamm_score', 'f_score', 'precision', 'recall']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'hamm_score': hamm_score, 'f_score': fscore, 'precision': precision, 'recall': recall})
                                
                            
                            
                            test_writer.add_summary(summary, i)
                            verify_dir_exists('./results/')
                            # with open('./results/predictions_test_graph.txt', 'a+') as file:
                                # file.write('%s\tPrediction:%s\nActual:%s\n' % (str(datetime.now()), reports['prediction'], reports['actual_labels']))
                            
                        start_temp = time.time()
                        _, reports = sess.run([train_step, self.reports], feed_dict={self.net.is_training:1})
                        
                        #calculating precision, recall, fscore and hamming score
                        hamm_score = hamming_loss(reports['actual_labels'],reports['prediction'])
                        # fbeta_val = fbeta_score(reports['actual_labels'],reports['prediction'], beta=beta,average='samples')
                        precision, recall, fscore, _ = precision_recall_fscore_support(reports['actual_labels'],reports['prediction'],average='samples')
                        # assign_op_prec = prec.assign(precision)
                        # assign_op_rec = rec.assign(recall)
                        # assign_op_fscore = f_score.assign(fscore)
                        # assign_op_hamm = hamm_loss.assign(hamm_score)
                        
                        # summary, _, _, _, _ = sess.run([summary_merged, assign_op_prec, assign_op_rec, assign_op_fscore, assign_op_hamm], feed_dict={self.net.is_training:1})
                        summary = sess.run(summary_merged, feed_dict={self.net.is_training:1, prec:precision, rec: recall, f_score:fscore, hamm_loss:hamm_score})
                        total_training += time.time() - start_temp
                        i += 1
                        if ((i-1) % self.display_iter) == 0:
                            
                            train_writer.add_summary(summary, i-1)
                            total = time.time() - start_at
                            self.print_ext('Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (i-1, total_training/total, total_testing/total, time.time()-last_summary)) 
                            for key, value in reports.items():
                                if key != 'actual_labels' and key!='prediction' and key!='features' and key!='embed_features' and key!='prediction_weighted':
                                    self.print_ext('Training Step %d "%s" = ' % (i-1, key), value)
                                    
                            with open('./results/predictions_train.csv', 'a') as csvfile:
                                fieldnames = ['actual_labels', 'predictions']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'actual_labels': reports['actual_labels'], 'predictions': reports['prediction']})
                            with open('./results/scores_train.csv', 'a') as csvfile:
                                fieldnames = ['hamm_score', 'f_score', 'precision', 'recall']
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                writer.writerow({'hamm_score': hamm_score, 'f_score': fscore, 'precision': precision, 'recall': recall})
                            
                            last_summary = time.time()            
                        if (i-1) % 100 == 0:
                            total_training = 0.0
                            total_testing = 0.0
                            start_at = time.time()
                    if i % self.iterations_per_test == 0:
                        summary = sess.run(summary_merged, feed_dict={self.net.is_training:0, prec:precision, rec: recall, f_score:fscore, hamm_loss:hamm_score})
                        #if self.debug == False:
                        test_writer.add_summary(summary, i)
                        self.print_ext('Test Step %d Finished' % i)
                except KeyboardInterrupt as err:
                   self.print_ext('Training interrupted at %d' % i)
                   wasKeyboardInterrupt = True
                   raisedEx = err
                finally:
                    if i > 0:
                        # print_ext('Some error')
                        self.save_model(sess, saver, i)
                    self.print_ext('Training completed, starting cleanup!')
                    self.print_ext('Final testing with test set')
                    reports = sess.run(self.reports, feed_dict={self.net.is_training:-1}) 
                    # print self.test_idx
                    # print np.sum(self.test_idx)
                    # print np.shape(reports['prediction'])
                    # print np.sum(reports['actual_labels'],axis=0)
                    # print np.sum(reports['prediction'], axis=0)
                    hamm_score = hamming_loss(reports['actual_labels'],reports['prediction'])
                    precision, recall, fscore, _ = precision_recall_fscore_support(reports['actual_labels'],reports['prediction'],average='samples')
                    self.print_ext('Final accuracy:',fscore,' Precision:',precision,' Recall:',recall,' Hamming loss:',hamm_score)
                    # self.extract = True
                    # reports = sess.run(self.reports, feed_dict={self.net.is_training:1}) 
                    # precision, recall, fscore, _ = precision_recall_fscore_support(reports['actual_labels'],reports['prediction'],average='samples')
                    # self.print_ext('Final accuracy:',fscore,' Precision:',precision,' Recall:',recall,' Hamming loss:',hamm_score)
                    # print np.sum(reports['prediction'], axis=0)
                    coord.request_stop()
                    coord.join(threads)
                    self.print_ext('Cleanup completed!')
                    if wasKeyboardInterrupt:
                        raise raisedEx
                
                if self.kFold:
                    return sess.run([self.max_acc_test, self.net.global_step])
                else:
                    return fscore
        else:
            self.print_ext('Model "%s" already trained!' % self.model_name)
            # with tf.Graph().as_default():
            #dummy = tf.Variable(0)  # dummy variable !!!
            self.net.is_training = tf.placeholder(tf.int32, shape=())
            prec = tf.placeholder(tf.float32, shape=(), name='precision')
            rec = tf.placeholder(tf.float32, shape=(), name='recall')
            f_score = tf.placeholder(tf.float32, shape=(), name='f_score')
            hamm_loss = tf.placeholder(tf.float32, shape=(), name='hamm_loss')
            self.net.global_step = tf.Variable(i,name='global_step',trainable=False)
            input = self.create_data()
            self.net_constructor.create_network(self.net, input)
            self.create_loss_function()
            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer(), self.variable_initialization)
                self.print_ext('Starting threads')
                saver = tf.train.Saver()  # Gets all variables in `graph`.
                i = self.load_model(sess, saver)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                num = 2100
                features_mat = np.zeros((num,17))
                index_mat = np.zeros((num))
                class_mat = np.zeros((num))
                pred_labels = np.zeros((num,17))
                actual_labels = np.zeros((num,17))
                embed_features = np.zeros((num,413,256))
                first_gc_features = np.zeros((num,64,128))
                second_gc_features = np.zeros((num,32,64))
                fc_features = np.zeros((num,256))
                train_idx = self.train_idx
                val_idx = self.val_idx
                test_idx = self.test_idx
                for ind in range(0,num/self.train_batch_size): #processing the datapoints batchwise
                    reports = sess.run(self.reports, feed_dict={self.net.is_training:1})
                    # features_mat[(ind*self.train_batch_size):((ind+1)*self.train_batch_size),:] = reports['features'] #storing features
                    ind_s = ind*self.train_batch_size
                    ind_e = (ind+1)*self.train_batch_size
                    features_mat[ind_s:ind_e,:] = reports['features'] #storing features
                    index_mat[ind_s:ind_e] = reports['index']
                    class_mat[ind_s:ind_e] = reports['classes']
                    actual_labels[ind_s:ind_e,:] = reports['actual_labels']
                    pred_labels[ind_s:ind_e,:] = reports['prediction']
                    embed_features[ind_s:ind_e,:,:] = reports['embed_features']
                    first_gc_features[ind_s:ind_e,:,:] = reports['first_gc_features']
                    second_gc_features[ind_s:ind_e,:,:] = reports['second_gc_features']
                    fc_features[ind_s:ind_e,:] = reports['fc_features']
                    self.print_ext('Processing %d batch' % ind)
                sio.savemat('gcn_features_new2.mat', {'features':features_mat, 'index':index_mat, 'classes':class_mat, 'actual_labels': actual_labels, 'embed_features': embed_features, 'pred_labels':pred_labels,'fc_features':fc_features,'first_gc_features':first_gc_features,'second_gc_features':second_gc_features}) #saving
                sio.savemat('test_train_idx.mat',{'train_ind' : train_idx, 'val_ind': val_idx, 'test_ind': test_idx})
                hamm_score = hamming_loss(actual_labels, pred_labels)
                precision, recall, fscore, _ = precision_recall_fscore_support(actual_labels, pred_labels,average='samples')
                self.print_ext('Final accuracy:',fscore,' Precision:',precision,' Recall:',recall,' Hamming loss:',hamm_score)
                hamm_score = hamming_loss(actual_labels[test_idx,:], pred_labels[test_idx,:])
                precision, recall, fscore, _ = precision_recall_fscore_support(actual_labels[test_idx,:], pred_labels[test_idx,:],average='samples')
                print np.sum(test_idx)
                print np.sum(pred_labels[test_idx,:], axis=0)
                print np.sum(actual_labels, axis=0)
                self.print_ext('Final accuracy:',fscore,' Precision:',precision,' Recall:',recall,' Hamming loss:',hamm_score)
                for i in range(0,17):
                    cm = confusion_matrix(actual_labels[:,i], pred_labels[:,i])
                    print(cm)
                
                coord.request_stop()
                coord.join(threads)
                self.print_ext('Cleanup completed!')
                
            
            if self.kFold:
                return self.get_max_accuracy()
            elif self.extract == False:
                return fbeta_val
            else:
                return 0

