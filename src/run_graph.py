import graphcnn.setup.ucmerced as sc
from graphcnn.experiment_multilabel import *


dataset = sc.load_ucmerced_dataset() # to load the data

class GraphExperiment():
    def create_network(self, net, input):
        net.create_network(input)
        net.make_embedding_layer(256) #takes input of any cardinality and produces output of fixed size
        net.make_dropout_layer()
        net.make_graphcnn_layer(128) 
        net.make_dropout_layer()
        net.make_graph_embed_pooling(no_vertices=64) 
        net.make_dropout_layer()
            
        net.make_graphcnn_layer(64) 
        net.make_dropout_layer()
        net.make_graph_embed_pooling(no_vertices=32) 
        net.make_dropout_layer()
        
        net.make_fc_layer(256) 
        net.make_dropout_layer()
        net.make_fc_layer(17, name='final', with_bn=False, with_act_func = False)
        
exp = GraphCNNExperiment('Graph', 'graph', GraphExperiment()) 

exp.num_iterations = 250000 #num of iterations
exp.optimizer = 'momentum' #optimizer
exp.debug = True #verbose mode
        
exp.preprocess_data(dataset) #preprocess data (zero pad adjacency matrix,etc)
beta = 1.0  #for weighing precision and recall in f-score

acc = exp.run_experiments(beta=beta, threshold = 0.5) #run experiment, pass threshold for classification
print_ext('Accuracy(f-score): %.2f' % acc)
