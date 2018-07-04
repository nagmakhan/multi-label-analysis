from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, fbeta_score, confusion_matrix, precision_recall_fscore_support
from skmultilearn.problem_transform import BinaryRelevance # to help stratify while sampling
import scipy.io
import numpy as np
import time


## Loading data
print("Loading data")


# Raw features - do data pre-process
features = scipy.io.loadmat('dataset/features.mat')
features = features['features']
features = features['val']
features = features[0]

# labels
labels = scipy.io.loadmat('dataset/UCMERCED/multilabels/LandUse_multilabels.mat')
labels = labels['labels']
labels = np.squeeze(np.transpose(labels,(1,0)))

# ## Data pre-process - mean of all node features for feeding to conventional classiifers which can not deal with graphs
print("Pre-processing data")
graph_size = np.array([s.shape[0] for s in features]).astype(np.int64)   
largest_graph = max(graph_size)
features_mat = np.zeros((np.shape(features)[0], largest_graph, np.shape(features[0])[1]))
for i in range(np.shape(features)[0]):
    features_mat[i,:,:] = np.pad(features[i].astype(np.float32), ((0,largest_graph-features[i].shape[0]), (0, 0)), 'constant', constant_values=(0))
    
features = np.mean(features_mat, axis=1) #final mean features


## Analysis for GCN
print('Analysis for GCN..')
index = scipy.io.loadmat('test_train_idx.mat')
feat = scipy.io.loadmat('gcn_features.mat')
predictions = feat['pred_labels']
train_ind = np.squeeze(index['train_ind'])
test_ind = np.squeeze(index['test_ind'])
val_ind = np.squeeze(index['val_ind'])

X_test = features[test_ind,:]
y_test = labels[test_ind, :]
print(np.sum(labels,axis=0))
print (np.sum(predictions,axis=0))
# print y_test
predictions = np.squeeze(predictions[test_ind,:])
print (np.sum(predictions,axis=0))

# print predictions
for i in range(0,17):
    print(confusion_matrix(y_test[:,i], predictions[:,i]))


# # # score calculation
print("Score calculation..")
precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='macro')
print("Precision macro:%f" % precision)
print("Recall macro:%f" % recall)
print("F-score macro:%f" % fscore)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='samples')

print("Precision samples:%f" % precision)
print("Recall samples:%f" % recall)
print("F-score samples:%f" % fscore)



# using same index set for train, test and val as GCN for the other classifiers 
X_train = features[train_ind,:]
y_train = labels[train_ind, :]
X_val = features[val_ind,:]
y_val = labels[val_ind, :]
X_test = features[test_ind,:]
y_test = labels[test_ind, :]




## Multi-label classification
## Using KNN

from skmultilearn.adapt import MLkNN

# for last-layer, k=13
classifier = MLkNN(k=13) #tuned


# train
start_at = time.time()
print("Training classifier KNN")
classifier.fit(X_train, y_train)
print("Training finished in time %g seconds" % (time.time()-start_at))


# # # predict
print("Predicting")
predictions = (classifier.predict(X_test))
predictions = predictions.todense();
predictions_val = ((classifier.predict(X_val)))
predictions_val = predictions_val.todense();
predictions_train = ((classifier.predict(X_train)))
predictions_train = predictions_train.todense();
predictions_all = np.vstack((predictions, predictions_val, predictions_train))
prob_test = ((classifier.predict_proba(X_test)))
prob_test = prob_test.todense()
prob_val = ((classifier.predict_proba(X_val)))
prob_val = prob_val.todense()
prob_train = ((classifier.predict_proba(X_train)))
prob_train = prob_train.todense()
prob = np.vstack((prob_test,prob_val,prob_train))


# # # score calculation
print("Score calculation..")

precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='macro')
print("Precision macro:%f" % precision)
print("Recall macro:%f" % recall)
print("F-score macro:%f" % fscore)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='samples')

print("Precision samples:%f" % precision)
print("Recall samples:%f" % recall)
print("F-score samples:%f" % fscore)

# ## Using Gaussian NB
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
start_at=time.time()
print("Training classifier gaussian NB")
classifier.fit(X_train, y_train)
print("Training finished in time %g seconds" % (time.time()-start_at))

# predict
print("Predicting")
predictions = classifier.predict(X_test)
predictions = (classifier.predict(X_test))
predictions = predictions.todense();
predictions_val = ((classifier.predict(X_val)))
predictions_val = predictions_val.todense();
predictions_train = ((classifier.predict(X_train)))
predictions_train = predictions_train.todense();
predictions_all = np.vstack((predictions, predictions_val, predictions_train))
prob_test = ((classifier.predict_proba(X_test)))
prob_test = prob_test.todense()
prob_val = ((classifier.predict_proba(X_val)))
prob_val = prob_val.todense()
prob_train = ((classifier.predict_proba(X_train)))
prob_train = prob_train.todense()
prob = np.vstack((prob_test,prob_val,prob_train))

# score calculation
print("Score calculation..")


precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='macro')
print("Precision macro:%f" % precision)
print("Recall macro:%f" % recall)
print("F-score macro:%f" % fscore)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='samples')

print("Precision samples:%f" % precision)
print("Recall samples:%f" % recall)
print("F-score samples:%f" % fscore)


from sklearn.svm import SVC


classifier = BinaryRelevance(classifier = SVC(C=2.2, probability=True))

# train
start_at = time.time()
print("Training classifier SVC with Binary Relevance")
classifier.fit(X_train, y_train)
print("Training finished in time %g seconds" % (time.time()-start_at))

# predict
print("Predicting")
predictions = (classifier.predict(X_test))
predictions = predictions.todense();
predictions_val = ((classifier.predict(X_val)))
predictions_val = predictions_val.todense();
predictions_train = ((classifier.predict(X_train)))
predictions_train = predictions_train.todense();
predictions_all = np.vstack((predictions, predictions_val, predictions_train))
prob_test = ((classifier.predict_proba(X_test)))
prob_test = prob_test.todense()
prob_val = ((classifier.predict_proba(X_val)))
prob_val = prob_val.todense()
prob_train = ((classifier.predict_proba(X_train)))
prob_train = prob_train.todense()
prob = np.vstack((prob_test,prob_val,prob_train))



# score calculation
print("Score calculation..")
precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='macro')
print("Precision macro:%f" % precision)
print("Recall macro:%f" % recall)
print("F-score macro:%f" % fscore)

precision, recall, fscore, _ = precision_recall_fscore_support(y_test,predictions ,average='samples')

print("Precision samples:%f" % precision)
print("Recall samples:%f" % recall)
print("F-score samples:%f" % fscore)
