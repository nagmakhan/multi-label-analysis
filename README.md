# multi-label-analysis

This set of codes implements multi-label classification of images using graph convolution network (GCN). In this work, it is implemented for VHR airborne images using the multi-label annotated UCMERCED dataset (source: <a href="https://ieeexplore.ieee.org/document/8089668/">Chaudhuri et al</a>) but it is a generic framework. The GCN used in this framework is inpired from <a href="https://ieeexplore.ieee.org/document/7979525/">Such et al</a>. The codes are written using the TensorFlow framework (version 1.2.1) in Python 2.7.12. The input needed for the code is adjacency matrices of the graph, node features and label set. 

To implement the code:
<ol>
<li>Check the gpu being used and amount of gpu in <b>src/graphcnn/experiment_multilabel.py</b> file (look for this code snippet in the _init_ function, here 0.2 fraction of gpu0 is being used) <br>
 &nbsp &nbsp self.config.gpu_options.per_process_gpu_memory_fraction = 0.2 <br>
 &nbsp &nbsp os.environ["CUDA_VISIBLE_DEVICES"] = '0’</li> 

<li>If needed change the path of snapshots and summary folders in ‘run’ function of <b>src/graphcnn/experiment_multilabel.py</b> by changing the ‘path’ variable

<li> Change the mat file locations in <b>src/graphcnn/setup/ucmerced.py</b> (the mat files holding the adjacency matrices and node features) i.e.   <br> 
     &nbsp &nbsp dataset= scipy.io.loadmat('/path_to_mat_file/new_dataset.mat')</li>


<li> While in src folder, run the <b>run_graph.py</b> file (for terminal based, type ‘python run_graph.py’ in terminal) </li> </ol>

The various useful files and their details are:
<ol>
<li> <b> src/graphcnn/setup/ucmerced.py </b> - the file which loads the mat file and sends it to run_graph.py which is used later. You need to load the graph’s adjacency matrix, features and training labels here. </li>

<li> <b> src/run_graph.py </b> - the file to be run, from which load_ucmerced_data, preprocess_data and experiment_multilabel functions are called. Edit this if you want to change architecture and other misc parameters.</li>

<li> <b> src/graphcnn/experiment_multilabel.py </b> - the main file, having the main functions called from run_graph.py. Edit the various functions, main function is run_experment from which others are called like create_test_train, create_data and run. </li>

<li> <b> src/graphcnn/network.py, Graph-CNN/src/graphcnn/layer.py </b> - the back end files containing the details about graph cnn layers. DO NOT edit unless you want to make any change in architecture </li> 

<li> <b> src/multi_label.py</b> - to run the multi label comparison experiments with SVC, MLkNN and GaussianNB. Loads GCN’s train, test and val data and uses the same here.</li>
</ol>
