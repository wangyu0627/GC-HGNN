import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
import dgl
from collections import defaultdict
from torch_geometric.data import HeteroData

def TensorToDGL(g_list):
    gs = []
    for tensor in g_list:
        tensor = tensor.coalesce()
        indices = tensor.indices()
        values = tensor.values()
        graph = dgl.graph((indices[0], indices[1]), num_nodes=tensor.shape[0])
        graph.edata['edge_weight'] = values
        gs.append(graph)
    return gs

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def process_data_in_pyg(neigs):
    d = defaultdict(dict)
    metapaths = []
    for mp_i, nei1 in enumerate(neigs):
        dst_array_concat = np.concatenate(nei1)
        src_array_concat = []
        for src_id, dst_array in enumerate(nei1):
            src_array_concat.extend([src_id] * len(dst_array))
        src_array_concat = np.array(src_array_concat)
        src_name = f"target"
        dst_name = f"dst_{mp_i}"
        relation = f"relation_{mp_i}"
        d[(src_name, relation + "-->", dst_name)]["edge_index"] = th.LongTensor(np.vstack([src_array_concat, dst_array_concat]))
        metapaths.append((src_name, relation + "-->", dst_name))
        d[(dst_name, "<--" + relation, src_name)]["edge_index"] = th.LongTensor(np.vstack([dst_array_concat, src_array_concat]))
        metapaths.append((dst_name, "<--" + relation, src_name))
    g = HeteroData(d)
    return g, metapaths

def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    
    # neighbor sample 
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_s = [th.LongTensor(i) for i in nei_s]
    
    # feat
    feat_p = sp.load_npz(path + "p_feat.npz")
    # feat_p = torch.load('../data/emb/acm/psp.npz')
    # feat_a = sp.load_npz(path + "a_feat.npz")
    feat_a = sp.eye(type_num[1])
    feat_s = sp.eye(type_num[2])
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    
    # meta-path graph
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    
    # pos
    # la_pos = sp.load_npz(path + "la_pos.npz")
    # sa_pos = sp.load_npz(path + "sa_pos.npz")
    pos = sp.load_npz(path + "pos.npz")
    # la_pos = sparse_mx_to_torch_sparse_tensor(la_pos)
    # sa_pos = sparse_mx_to_torch_sparse_tensor(sa_pos)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    # set
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    meta_paths = {"paper": [["pa", "ap"], ["ps", "sp"]]}
    predict_key = 'paper'
    return [nei_a, nei_s], predict_key, meta_paths, [pap, psp], [feat_a, feat_p, feat_s], pos, label, train, val, test
    #     [la_pos, sa_pos],

def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    
    # neighbor sample 
    nei_d = np.load(path + "nei_d.npy", allow_pickle=True)
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_w = np.load(path + "nei_w.npy", allow_pickle=True)
    nei_d = [th.LongTensor(i) for i in nei_d]
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_w = [th.LongTensor(i) for i in nei_w]
    
    # feat
    # feat_m = torch.load('../data/emb/freebase/mam.npz')
    # feat_d = torch.load('../data/emb/freebase/dmd.npz')
    # feat_a = torch.load('../data/emb/freebase/ama.npz')
    # feat_w = torch.load('../data/emb/freebase/wmw.npz')
    feat_m = sp.eye(type_num[0])
    feat_d = sp.eye(type_num[1])
    feat_a = sp.eye(type_num[2])
    feat_w = sp.eye(type_num[3])
    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_w = th.FloatTensor(preprocess_features(feat_w))
    # Because none of M, D, A or W has features, we assign one-hot encodings to all of them.
    
    # meta-path graph
    mam = sp.load_npz(path + "mam.npz")
    mdm = sp.load_npz(path + "mdm.npz")
    mwm = sp.load_npz(path + "mwm.npz")
    mam = sparse_mx_to_torch_sparse_tensor(normalize_adj(mam))
    mdm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mdm))
    mwm = sparse_mx_to_torch_sparse_tensor(normalize_adj(mwm))
    
    # pos
    pos = sp.load_npz(path + "pos.npz")
    # la_pos = sp.load_npz(path + "la_pos.npz")
    sa_pos = sp.load_npz(path + "sa_pos.npz")
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    # la_pos = sparse_mx_to_torch_sparse_tensor(la_pos)
    sa_pos = sparse_mx_to_torch_sparse_tensor(sa_pos)
    
    # set
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    meta_paths = {"movie": [["ma", "am"], ["md", "dm"], ["mw", "wm"]]}
    predict_key = 'movie'

    return [nei_d, nei_a, nei_w], predict_key, meta_paths, [mdm, mam, mwm], [feat_a, feat_d, feat_m, feat_w], pos, label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    
    # neighbor sample 
    nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    nei_p = [th.LongTensor(i) for i in nei_p]
    
    # feat
    feat_t = torch.load('../data/emb/dblp/tpt.npz')
    feat_a = sp.eye(type_num[0])
    feat_p = sp.eye(type_num[1])
    feat_c = sp.eye(type_num[3])
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_c = th.FloatTensor(preprocess_features(feat_c))
    
    # meta-path graph
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    apa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apa))
    apcpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(apcpa))
    aptpa = sparse_mx_to_torch_sparse_tensor(normalize_adj(aptpa))
    
    # pos
    # la_pos = sp.load_npz(path + "la_pos.npz")
    sa_pos = sp.load_npz(path + "sa_pos.npz")
    # pos = sp.load_npz(path + "pos.npz")
    # la_pos = sparse_mx_to_torch_sparse_tensor(la_pos)
    sa_pos = sparse_mx_to_torch_sparse_tensor(sa_pos)
    # pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    # set
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    meta_paths = {"author": [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]]}
    predict_key = 'author'
    return [nei_p], predict_key, meta_paths, [apa, apcpa, aptpa], [feat_a, feat_c, feat_p, feat_t], sa_pos, label, train, val, test


def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)
    
    # neighbor sample 
    nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    nei_a = [th.LongTensor(i) for i in nei_a]
    nei_r = [th.LongTensor(i) for i in nei_r]
    
    # feat 
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    # feat_p = torch.load('../data/emb/aminer/prp.npz')
    feat_a = torch.load('../data/emb/aminer/apa.npz')
    # feat_r = torch.load('../data/emb/aminer/rpr.npz')
    feat_p = sp.eye(type_num[0])
    # feat_a = sp.eye(type_num[1])
    feat_r = sp.eye(type_num[2])
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    # feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_r = th.FloatTensor(preprocess_features(feat_r))
    
    # meta-path graph
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    prp = sparse_mx_to_torch_sparse_tensor(normalize_adj(prp))
    
    # pos 
    pos = sp.load_npz(path + "pos.npz")
    la_pos = sp.load_npz(path + "la_pos.npz")
    sa_pos = sp.load_npz(path + "sa_pos.npz")
    la_pos = sparse_mx_to_torch_sparse_tensor(la_pos)
    sa_pos = sparse_mx_to_torch_sparse_tensor(sa_pos)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    
    # set
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    
    meta_paths = {"paper": [["pa", "ap"], ["pr", "rp"]]}
    predict_key = 'paper'
    return [nei_a, nei_r], predict_key, meta_paths, [pap, prp], [feat_a, feat_p, feat_r], pos, label, train, val, test




def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
        
    g, metapaths = process_data_in_pyg(data[0])
    return data, g, metapaths
