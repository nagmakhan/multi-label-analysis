import graphcnn.setup.ucmerced as sc
from graphcnn.experiment_multilabel import *

 os.environ["CUDA_VISIBLE_DEVICES"] = '1'

dataset = sc.load_ucmerced_dataset()

class GraphExperiment():
    def create_network(self, net, input):
        net.create_network(input)
        net.make_embedding_layer(256) #takes input of any cardinality and produces output of fixed size
        net.make_dropout_layer()
        net.make_graphcnn_layer(128) #earlier 64, then 128 for best performance
        net.make_dropout_layer()
        net.make_graph_embed_pooling(no_vertices=64) #earlier 32, then 64 for best performance
        net.make_dropout_layer()
            
        net.make_graphcnn_layer(64) #earlier 32, then 64 for best performance
        net.make_dropout_layer()
        net.make_graph_embed_pooling(no_vertices=32) #earlier 8, then 16 for best performance, 32 for later best
        net.make_dropout_layer()
        
        net.make_fc_layer(256) #later best, earlier 256
        net.make_dropout_layer()
        net.make_fc_layer(17, name='final', with_bn=False, with_act_func = False)
        
exp = GraphCNNExperiment('Graph', 'graph', GraphExperiment())

exp.num_iterations = 250000
exp.optimizer = 'momentum'
exp.debug = True
        
exp.preprocess_data(dataset)
beta = 1.0

acc = exp.run_experiments(beta=beta, threshold = 0.5)
print_ext('Accuracy(f-score): %.2f' % acc)
