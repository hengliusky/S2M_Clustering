import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.nn import Parameter
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
import metrics as mt
import scipy.sparse as sp
from preprocessing import sparse_to_tuple
from visualization import t_sne
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from layer import SuperGAT
from collections import Counter

#np.set_printoptions(threshold=np.inf)

def norm_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q ** ((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q


def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    confidence1 = np.zeros((q.shape[0],))
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i, a[i, -1]]
        confidence2[i] = q[i, a[i, -2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            conf_indices.append(i)
        else:
            unconf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices


def target_distribution(q):
    p = torch.nn.functional.one_hot(torch.argmax(q, dim=1), q.shape[1]).to(dtype=torch.float32)
    return p


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)

        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print(
            'ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write(
            'ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)



class ReDGAE(nn.Module):
    def __init__(self, **kwargs):
        super(ReDGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        self.alpha = kwargs['alpha']
        self.gamma = kwargs['gamma']
        self.heads = kwargs['heads']
        self.dropout = kwargs['dropout']
        self.attention_type = kwargs['attention_type']
        self.neg_sample_ratio = kwargs['neg_sample_ratio']
        self.edge_sampling_ratio = kwargs['edge_sampling_ratio']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh
        # layers
        self.gcn_add1 = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_add2 = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x: x)
        self.gcn_rm1 = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_rm2 = GraphConvSparse(self.num_neurons, self.embedding_size, activation=lambda x: x)
        self.gat_add1 = SuperGAT(self.num_features, self.num_neurons,
                                heads=self.heads, dropout=self.dropout, concat=True, attention_type=self.attention_type,
                                neg_sample_ratio=self.neg_sample_ratio, edge_sample_ratio=self.edge_sampling_ratio)
        self.gat_add2 = SuperGAT(self.num_neurons * self.heads, self.embedding_size,
                                 heads=self.heads, dropout=self.dropout, concat=False,
                                 attention_type=self.attention_type,
                                 neg_sample_ratio=self.neg_sample_ratio, edge_sample_ratio=self.edge_sampling_ratio)
        self.gat_rm1 = SuperGAT(self.num_features, self.num_neurons,
                                 heads=self.heads, dropout=self.dropout, concat=True,
                                 attention_type=self.attention_type,
                                 neg_sample_ratio=self.neg_sample_ratio, edge_sample_ratio=self.edge_sampling_ratio)
        self.gat_rm2 = SuperGAT(self.num_neurons * self.heads, self.embedding_size,
                                 heads=self.heads, dropout=self.dropout, concat=False,
                                 attention_type=self.attention_type,
                                 neg_sample_ratio=self.neg_sample_ratio, edge_sample_ratio=self.edge_sampling_ratio)
        self.assignment_add = ClusterAssignment(self.nClusters, self.embedding_size, self.alpha)
        self.assignment_rm = ClusterAssignment(self.nClusters, self.embedding_size, self.alpha)
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.label_concession = 0.02
    #For the pre-training part, the pre-training model has been provided in . /results
    def pretrain(self, adj_add, features_add, adj_add_label,weight_tensor_add,edge_index_add,
                 adj_rm, features_rm, adj_rm_label, weight_tensor_rm,edge_index_rm,
                 optimizer, epochs, lr):
        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.001)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr)
        epoch_bar = tqdm(range(epochs))
        km_add = KMeans(n_clusters=self.nClusters, n_init=20)
        km_rm = KMeans(n_clusters=self.nClusters, n_init=20)
        for _ in epoch_bar:
            opti.zero_grad()
            z_add_o = self.encode_add(features_add, adj_add)
            z_add_att,cache_add = self.gat_encode_add(features_add,edge_index_add)
            z_add = z_add_o + 0.5 * z_add_att
            x_add = self.decode(z_add)

            z_rm_o = self.encode_rm(features_rm, adj_rm)
            z_rm_att,cache_rm = self.gat_encode_rm(features_rm,edge_index_rm)
            z_rm = z_rm_o + 0.5 * z_rm_att
            x_rm = self.decode(z_rm)
            loss = F.binary_cross_entropy(x_add.view(-1), adj_add_label.to_dense().view(-1), weight=weight_tensor_add) + F.binary_cross_entropy(x_rm.view(-1), adj_rm_label.to_dense().view(-1), weight=weight_tensor_rm)
            print(loss)
            loss.backward()
            opti.step()
        km_add.fit(z_add.detach().numpy())
        km_rm.fit(z_rm.detach().numpy())
        centers_add = torch.tensor(km_add.cluster_centers_, dtype=torch.float, requires_grad=True)
        centers_rm = torch.tensor(km_rm.cluster_centers_, dtype=torch.float, requires_grad=True)
        self.assignment_add.state_dict()["cluster_centers"].copy_(centers_add)
        self.assignment_rm.state_dict()["cluster_centers"].copy_(centers_rm)
    #Loss functions, including reconstruction loss, similarity loss and clustering loss
    def loss(self, q_add, p_add, x_add, adj_add_label, weight_tensor_add,cache_add,
            q_rm, p_rm, x_rm, adj_rm_label, weight_tensor_rm,cache_rm):
        loss_recons = F.binary_cross_entropy(x_add.view(-1), adj_add_label.to_dense().view(-1), weight=weight_tensor_add) + F.binary_cross_entropy(x_rm.view(-1), adj_rm_label.to_dense().view(-1), weight=weight_tensor_rm)
        loss_clus = self.kl_loss(torch.log(q_add), p_add) + self.kl_loss(torch.log(q_rm), p_rm)
        loss_att = SuperGAT.get_supervised_attention_loss(cache_add) + SuperGAT.get_supervised_attention_loss(cache_rm)
        # print("##########################")
        # print(loss_att,loss_recons,loss_clus)
        loss = 0.5 * loss_recons + self.gamma * loss_clus + loss_att
        return loss, loss_recons, loss_clus

    #The training part of the model
    def train(self, adj_add_norm, features_add, adj_add, adj_add_label,weight_tensor_add, norm_add,edge_index_add,
              adj_rm_norm, features_rm, adj_rm, adj_rm_label, weight_tensor_rm, norm_rm,edge_index_rm,
              y,optimizer, epochs, lr, beta1, beta2,save_path, dataset):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'),False)
        #print("adj add len"+str(len(adj_add_label._values())))
        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        epoch_bar = tqdm(range(epochs))
        epoch_add_stable = 0
        epoch_rm_stable = 0
        epoch_bar = tqdm(range(epochs))

        km_add = KMeans(n_clusters=self.nClusters, n_init=20)
        km_rm = KMeans(n_clusters=self.nClusters, n_init=20)
        print('Training......')
        beta1_add = beta1
        beta2_add = beta2
        beta1_rm = beta1
        beta2_rm = beta2
        #Initialize the clustering center
        with torch.no_grad():
            emb_add_o = self.encode_add(features_add, adj_add_norm)
            emb_add_att,_ = self.gat_encode_add(features_add, edge_index_add)
            emb_add = emb_add_o + 0.5 * emb_add_att
            km_add.fit(emb_add.detach().numpy())
            centers_add = torch.tensor(km_add.cluster_centers_, dtype=torch.float, requires_grad=True)
            self.assignment_add.state_dict()["cluster_centers"].copy_(centers_add)


            emb_rm_o = self.encode_rm(features_rm, adj_rm_norm)
            emb_rm_att,_ = self.gat_encode_rm(features_rm, edge_index_rm)
            emb_rm = emb_rm_o + 0.5 * emb_rm_att
            km_rm.fit(emb_rm.detach().numpy())
            centers_rm = torch.tensor(km_rm.cluster_centers_, dtype=torch.float, requires_grad=True)
        #print(y_add_pred)
        previous_unconflicted_add = []
        previous_conflicted_add = []
        unconflicted_ind_add = []
        conflicted_ind_add = []
        previous_unconflicted_rm = []
        previous_conflicted_rm = []
        unconflicted_ind_rm = []
        conflicted_ind_rm = []

        for epoch in epoch_bar:
            opti.zero_grad()
            emb_add_o = self.encode_add(features_add, adj_add_norm)
            #t_sne(emb_add_o.detach().numpy(), y, 2708, True)
            emb_add_att,cache_add = self.gat_encode_add(features_add,edge_index_add)
            # print("emb_add shape" + str(emb_add.shape))
            emb_rm_o = self.encode_rm(features_rm, adj_rm_norm)
            emb_rm_att,cache_rm = self.gat_encode_rm(features_rm, edge_index_rm)
            # print("emb_rm shape" + str(emb_rm.shape))

            emb_add = emb_add_o + 0.5 * emb_add_att
            #Pseudo-labeling of prediction nodes
            emb_add_s = self.assignment_add.cluster_centers[self.predict(emb_add)]
            emb_add = emb_add + 0.2 * emb_add_s
            emb_rm = emb_rm_o + 0.5 * emb_rm_att
            emb_rm_s = self.assignment_rm.cluster_centers[self.predict(emb_rm)]
            emb_rm = emb_rm + 0.2 * emb_rm_s
            #Original probability distribution
            q_add = self.assignment_add(emb_add)
            q_rm = self.assignment_rm(emb_rm)
            #Target probability distribution
            if epoch % 15 == 0:
                p_add = target_distribution(q_add.detach())
                p_rm = target_distribution(q_rm.detach())
            #Inner product decoder
            x_add = self.decode(emb_add)
            x_rm = self.decode(emb_rm)
            if epoch % 15 == 0:
                #Filtering high-confidence nodes
                unconflicted_ind_add, conflicted_ind_add = generate_unconflicted_data_index(emb_add.detach().numpy(),
                                                                                    self.assignment_add.cluster_centers.detach().numpy(),
                                                                               beta1_add, beta2_add)
                unconflicted_ind_rm, conflicted_ind_rm = generate_unconflicted_data_index(emb_rm.detach().numpy(),
                                                                                          self.assignment_rm.cluster_centers.detach().numpy(),
                                                                                          beta1_rm, beta2_rm)
                # print(unconflicted_ind_add, conflicted_ind_add)
                # print(unconflicted_ind_rm, conflicted_ind_rm)
                if epoch == 0:
                    adj_add, adj_add_label, weight_tensor_add = self.update_graph(adj_add, y, emb_add, conflicted_ind_add)
                    edge_index_add = torch.tensor(np.vstack((adj_add_label._indices()[0], adj_add_label._indices()[1])),
                                                  dtype=torch.int32)
                    adj_rm, adj_rm_label, weight_tensor_rm = self.update_graph(adj_rm, y, emb_rm,conflicted_ind_rm)
                    edge_index_rm = torch.tensor(np.vstack((adj_rm_label._indices()[0], adj_rm_label._indices()[1])),
                                                  dtype=torch.int32)
                    # print("update adj add len" + str(len(adj_add_label._values())))
                    #print(adj_add_label)
            if len(previous_conflicted_add) < len(conflicted_ind_add):
                emb_add_conf = emb_add[conflicted_ind_add]
                p_add_conf = p_add[conflicted_ind_add]
                q_add_conf = q_add[conflicted_ind_add]
                previous_conflicted_add = conflicted_ind_add
                previous_unconflicted_add = unconflicted_ind_add
            else:
                epoch_add_stable += 1
                emb_add_conf = emb_add[previous_conflicted_add]
                p_add_conf = p_add[previous_conflicted_add]
                q_add_conf = q_add[previous_conflicted_add]
            if epoch_add_stable >= 15:
                epoch_add_stable = 0
                beta1_add = beta1_add * 0.95
                beta2_add = beta2_add * 0.85

            if len(previous_conflicted_rm) < len(conflicted_ind_rm):
                emb_rm_conf = emb_rm[conflicted_ind_rm]
                p_rm_conf = p_rm[conflicted_ind_rm]
                q_rm_conf = q_rm[conflicted_ind_rm]
                previous_conflicted_rm = conflicted_ind_rm
                previous_unconflicted_rm = unconflicted_ind_rm
            else:
                epoch_rm_stable += 1
                emb_rm_conf = emb_rm[previous_conflicted_rm]
                p_rm_conf = p_rm[previous_conflicted_rm]
                q_rm_conf = q_rm[previous_conflicted_rm]
            if epoch_rm_stable >= 15:
                epoch_rm_stable = 0
                beta1_rm = beta1_rm * 0.95
                beta2_rm = beta2_rm * 0.85
            #Updating clustering-oriented structural supervision information
            if epoch % 20 == 0 and epoch <= 120:
                adj_add, adj_add_label, weight_tensor_add = self.update_graph(adj_add, y, emb_add, conflicted_ind_add)
                edge_index_add = torch.tensor(np.vstack((adj_add_label._indices()[0], adj_add_label._indices()[1])),
                                              dtype=torch.int32)
                adj_rm, adj_rm_label, weight_tensor_rm = self.update_graph(adj_rm, y, emb_rm, conflicted_ind_rm)
                edge_index_rm = torch.tensor(np.vstack((adj_rm_label._indices()[0], adj_rm_label._indices()[1])),
                                             dtype=torch.int32)

            loss, _, _ = self.loss(q_add_conf, p_add_conf, x_add, adj_add_label, weight_tensor_add,cache_add,
                                   q_rm_conf, p_rm_conf, x_rm, adj_rm_label, weight_tensor_rm,cache_rm)
            y_pred = self.predict(emb_add)
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            #Cross-encoding component
            if epoch % 10 == 0 and epoch <= 200:
                print("START CROSSING")
                x_add,x_rm = x_rm,x_add
                adj_add_norm, adj_rm_norm = adj_rm_norm, adj_add_norm
                adj_add_norm = adj_add_norm.to_dense()
                adj_rm_norm = adj_rm_norm.to_dense()
                adj_add_norm = (1 - self.label_concession) * adj_add_norm + self.label_concession * adj_add_norm * x_add.detach()
                adj_rm_norm = (1 - self.label_concession) * adj_rm_norm + self.label_concession * adj_rm_norm * x_rm.detach()

                idx_add = torch.nonzero(adj_add_norm).T
                data_add = adj_add_norm[idx_add [0], idx_add [1]]
                adj_add_norm = torch.sparse_coo_tensor(idx_add, data_add, adj_add_norm.shape,requires_grad=False)  # 转换成COO矩阵
                idx_rm = torch.nonzero(adj_rm_norm).T
                data_rm = adj_rm_norm[idx_rm[0], idx_rm[1]]
                adj_rm_norm = torch.sparse_coo_tensor(idx_rm, data_rm, adj_rm_norm.shape,requires_grad=False)  # 转换成COO矩阵

                edge_index_add,edge_index_rm = edge_index_rm,edge_index_add
            loss.backward()
            opti.step()
            lr_s.step()
            if epoch == 199:
                t_sne(emb_add.detach().numpy(),y,2708,True)
    def predict(self, emb):
        with torch.no_grad():
            q = self.assignment_add(emb)
            out = np.argmax(q.detach().numpy(), axis=1)
        return out

    def predict_(self, emb):
        with torch.no_grad():
            q = self.assignment_rm(emb)
            out = np.argmax(q.detach().numpy(), axis=1)
        return out

    def encode_add(self, x_features, adj):
        hidden = self.gcn_add1(x_features, adj)
        self.embedded_add = self.gcn_add2(hidden, adj)
        return self.embedded_add

    def gat_encode_add(self, x, edge_index,**kwargs) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.gat_add1(x, edge_index, **kwargs)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=False)
        self.embedded_add = self.gat_add2(x, edge_index, **kwargs)
        self.cache_add = self.gat_add2.cache
        return self.embedded_add,self.cache_add

    def encode_rm(self, x_features, adj):
        hidden = self.gcn_rm1(x_features, adj)
        #print("hidden shape"+str(hidden.shape))
        self.embedded_rm = self.gcn_rm2(hidden, adj)
        #print("embedded_rm shape" + str(self.embedded_rm.shape))
        return self.embedded_rm

    def gat_encode_rm(self, x, edge_index,**kwargs) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.gat_rm1(x, edge_index, **kwargs)
        #print("x shape" + str(x.shape))
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=False)
        self.embedded_rm = self.gat_rm2(x, edge_index, **kwargs)
        #print("self.embedded_rm shape" + str(self.embedded_rm.shape))
        self.cache_rm = self.gat_rm2.cache
        return self.embedded_rm,self.cache_rm

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

    def generate_centers(self, emb_unconf, y_pred):
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.assignment_add.cluster_centers.detach().numpy())
        return indices[y_pred]

    def update_graph(self, adj, labels, emb, unconf_indices):
        y_pred = self.predict(emb)
        emb_unconf = emb[unconf_indices]
        adj = adj.tolil()
        idx = unconf_indices[self.generate_centers(emb_unconf, y_pred)]
        for i, k in enumerate(unconf_indices):
            adj_k = adj[k].tocsr().indices
            if not (np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]):
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, unconf_indices) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor
