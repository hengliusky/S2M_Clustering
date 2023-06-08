import numpy as np
import torch
import scipy.sparse as sp
from model_pubmed import ReDGAE
from datasets import format_data
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
import copy
import pickle
from visualization import t_sne

# Dataset Name
dataset = "Pubmed"
print("Pubmed dataset")
feas = format_data('pubmed', './data/Pubmed')
num_nodes = feas['features'].size(0)
num_features = feas['features'].size(1)
nClusters = 3
adj, features , labels = load_data('pubmed', './data/Pubmed')
A_pred = pickle.load(open(f'data/Pubmed/edge_probabilities/pubmed_graph_3_logits.pkl', 'rb'))
print(A_pred,A_pred.shape)
# Network parameters
alpha = 1.
gamma = 0.001

num_neurons = 32
embedding_size = 16
save_path = "./results/"
heads = 8
dropout = 0.6
attention_type = "scaled_dot_product"
neg_sample_ratio = 0.2
edge_sampling_ratio = 0.8
# Pretraining parameters
epochs_pretrain = 200
lr_pretrain = 0.005


# Clustering parameters
epochs_cluster = 200
lr_cluster = 0.005 #0.005
beta1 = 0.4
beta2 = 0.3 #0.3

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    #取出矩阵对角线上上三角(右上)部分
    # print(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    # print(edges)
    if remove_pct:
        print("############")
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        #排序，比第n_remove+1大小的元素放前面
        #概率前n_remove小的，要移除的边下标
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        #A_probs是预测的，edges已经存在的
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        # print(all_probs,all_probs.shape)
        #取后面n_add个概率大的节点
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        # print(e_index_2b_add)
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    #转化为对称矩阵
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

params = {'rm_pct':4, 'add_pct': 40}
adj_add = sample_graph_det(adj, A_pred, 0, params['add_pct'])
adj_rm = sample_graph_det(adj, A_pred, params['rm_pct'], 0)
print("adj len" + str(len(adj.data)))
print("adj_add len" + str(len(adj_add.data)))
print("adj_rm len" + str(len(adj_rm.data)))
# Data processing View 1
adj_add = adj_add - sp.dia_matrix((adj_add.diagonal()[np.newaxis, :], [0]), shape=adj_add.shape)
adj_add.eliminate_zeros()
adj_add_norm = preprocess_graph(adj_add)
features_add = sparse_to_tuple(features.tocoo())
num_features_add = features_add[2][1]
pos_weight_orig_add = float(adj_add.shape[0] * adj_add.shape[0] - adj_add.sum()) / adj_add.sum()
norm_add = adj_add.shape[0] * adj_add.shape[0] / float((adj_add.shape[0] * adj_add.shape[0] - adj_add.sum()) * 2)
adj_add_label = adj_add + sp.eye(adj_add.shape[0])
#print(adj_label)
adj_add_label = sparse_to_tuple(adj_add_label)
adj_add_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_add_norm[0].T), torch.FloatTensor(adj_add_norm[1]), torch.Size(adj_add_norm[2]))
adj_add_label = torch.sparse.FloatTensor(torch.LongTensor(adj_add_label[0].T), torch.FloatTensor(adj_add_label[1]), torch.Size(adj_add_label[2]))
features_add = torch.sparse.FloatTensor(torch.LongTensor(features_add[0].T), torch.FloatTensor(features_add[1]), torch.Size(features_add[2]))
weight_mask_orig_add = adj_add_label.to_dense().view(-1) == 1
weight_tensor_orig_add = torch.ones(weight_mask_orig_add.size(0))
weight_tensor_orig_add[weight_mask_orig_add] = pos_weight_orig_add
edge_index_add = torch.tensor(np.vstack((adj_add.tocoo().row, adj_add.tocoo().col)))

# Data processing View 2
adj_rm = adj_rm - sp.dia_matrix((adj_rm.diagonal()[np.newaxis, :], [0]), shape=adj_rm.shape)
adj_rm.eliminate_zeros()
adj_rm_norm = preprocess_graph(adj_rm)
features_rm = sparse_to_tuple(features.tocoo())
num_features_rm = features_rm[2][1]
pos_weight_orig_rm = float(adj_rm.shape[0] * adj_rm.shape[0] - adj_rm.sum()) / adj_rm.sum()
norm_rm = adj_rm.shape[0] * adj_rm.shape[0] / float((adj_rm.shape[0] * adj_rm.shape[0] - adj_rm.sum()) * 2)
adj_rm_label = adj_rm + sp.eye(adj_rm.shape[0])
#print(adj_label)
adj_rm_label = sparse_to_tuple(adj_rm_label)
adj_rm_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_rm_norm[0].T), torch.FloatTensor(adj_rm_norm[1]), torch.Size(adj_rm_norm[2]))
adj_rm_label = torch.sparse.FloatTensor(torch.LongTensor(adj_rm_label[0].T), torch.FloatTensor(adj_rm_label[1]), torch.Size(adj_rm_label[2]))
features_rm = torch.sparse.FloatTensor(torch.LongTensor(features_rm[0].T), torch.FloatTensor(features_rm[1]), torch.Size(features_rm[2]))
weight_mask_orig_rm = adj_rm_label.to_dense().view(-1) == 1
weight_tensor_orig_rm = torch.ones(weight_mask_orig_rm.size(0))
weight_tensor_orig_rm[weight_mask_orig_rm] = pos_weight_orig_rm
edge_index_rm = torch.tensor(np.vstack((adj_rm.tocoo().row, adj_rm.tocoo().col)))
# Training
network = ReDGAE(num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size,
                 nClusters=nClusters, activation="ReLU", alpha=alpha, gamma=gamma,heads=heads,dropout=dropout,attention_type=attention_type,
                 neg_sample_ratio=neg_sample_ratio,edge_sampling_ratio=edge_sampling_ratio,num_nodes=num_nodes)
# network.pretrain(adj_add_norm, features_add, adj_add_label,weight_tensor_orig_add,edge_index_add,
#                  adj_rm_norm, features_rm, adj_rm_label,weight_tensor_orig_rm,edge_index_rm,
#                  optimizer="Adam", epochs=epochs_pretrain, lr=lr_pretrain)
# torch.save(network.state_dict(), save_path + dataset + '/pretrain/my_model_w.pk')
network.train(adj_add_norm, features_add,adj_add, adj_add_label, weight_tensor_orig_add, norm_add,edge_index_add,
            adj_rm_norm, features_rm,adj_rm, adj_rm_label, weight_tensor_orig_rm, norm_rm,edge_index_rm,
            labels,optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset)
