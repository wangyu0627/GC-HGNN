import argparse
import sys


argv = sys.argv
dataset = argv[1]


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.18)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=1500)
    
    # The parameters of pretrain
    parser.add_argument('--use_pretrain', type=bool, default=False)
    parser.add_argument('--pre_epoch', type=int, default=30)
    parser.add_argument('--pre_embedding_dim', type=int, default=64)
    parser.add_argument('--pre_walk_length', type=int, default=10)
    parser.add_argument('--pre_context_size', type=int, default=3)
    parser.add_argument('--pre_walks_per_node', type=int, default=3)
    parser.add_argument('--pre_num_negative_samples', type=int, default=3)
    parser.add_argument('--pre_batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--La_pos_num', type=int, default=1)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=0.0005)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--fanout', type=int, default=1)
    parser.add_argument('--edge_dir', type=str, default='in')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--decoder', type=str, default='GCN')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--remask_rate', type=float, default=0.0)
    parser.add_argument('--num_remasking', type=int, default=1)
    parser.add_argument('--ntypes', type=dict, default={'author': 7167, 'paper': 1902, 'subject': 60})
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.5)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="freebase")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--nb_epochs', type=int, default=3000)
    
    # The parameters of pretrain
    parser.add_argument('--use_pretrain', type=bool, default=True)
    parser.add_argument('--pre_epoch', type=int, default=50)
    parser.add_argument('--pre_embedding_dim', type=int, default=64)
    parser.add_argument('--pre_walk_length', type=int, default=10)
    parser.add_argument('--pre_context_size', type=int, default=5)
    parser.add_argument('--pre_walks_per_node', type=int, default=3)
    parser.add_argument('--pre_num_negative_samples', type=int, default=3)
    parser.add_argument('--pre_batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--La_pos_num', type=int, default=10)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0)

    # model-specific parameters
    parser.add_argument('--fanout', type=int, default=1)
    parser.add_argument('--edge_dir', type=str, default='in')
    parser.add_argument('--encoder', type=str, default='GAT')
    parser.add_argument('--decoder', type=str, default='GAT')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--remask_rate', type=float, default=0.1)
    parser.add_argument('--num_remasking', type=int, default=1)
    parser.add_argument('--ntypes', type=dict, default={'actor': 33401, 'direct': 2502, 'movie': 3556, 'writer': 4459})
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--lam', type=float, default=0.6)

    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args

def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true", default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--nb_epochs', type=int, default=3000)
    
    # The parameters of pretrain
    parser.add_argument('--use_pretrain', type=bool, default=True)
    parser.add_argument('--pre_epoch', type=int, default=20)
    parser.add_argument('--pre_embedding_dim', type=int, default=64)
    parser.add_argument('--pre_walk_length', type=int, default=10)
    parser.add_argument('--pre_context_size', type=int, default=5)
    parser.add_argument('--pre_walks_per_node', type=int, default=3)
    parser.add_argument('--pre_num_negative_samples', type=int, default=3)
    parser.add_argument('--pre_batch_size', type=int, default=128)
    parser.add_argument('--pre_lr', type=float, default=0.005)
    parser.add_argument('--La_pos_num', type=int, default=100)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--fanout', type=int, default=1)
    parser.add_argument('--edge_dir', type=str, default='in')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--decoder', type=str, default='GAT')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.05)
    parser.add_argument('--remask_rate', type=float, default=0.05)
    parser.add_argument('--num_remasking', type=int, default=3)
    parser.add_argument('--ntypes', type=dict, default={'author': 4121, 'conference': 20, 'paper': 14328, 'term': 128})
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--lam', type=float, default=0.5) 

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--nb_epochs', type=int, default=1500)
    
    # The parameters of pretrain
    parser.add_argument('--use_pretrain', type=bool, default=True)
    parser.add_argument('--pre_epoch', type=int, default=10)
    parser.add_argument('--pre_embedding_dim', type=int, default=128)
    parser.add_argument('--pre_walk_length', type=int, default=10)
    parser.add_argument('--pre_context_size', type=int, default=5)
    parser.add_argument('--pre_walks_per_node', type=int, default=1)
    parser.add_argument('--pre_num_negative_samples', type=int, default=3)
    parser.add_argument('--pre_batch_size', type=int, default=256)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--La_pos_num', type=int, default=2)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_coef', type=float, default=0)
    
    # model-specific parameters
    parser.add_argument('--fanout', type=int, default=3)
    parser.add_argument('--edge_dir', type=str, default='out')
    parser.add_argument('--encoder', type=str, default='GAT')
    parser.add_argument('--decoder', type=str, default='GAT')
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.2)
    parser.add_argument('--remask_rate', type=float, default=0.1)
    parser.add_argument('--num_remasking', type=int, default=1)
    parser.add_argument('--feat_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.2)
    parser.add_argument('--ntypes', type=dict, default={'author': 128, 'paper': 6692, 'reference': 35890})
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.6)
    parser.add_argument('--lam', type=float, default=0.4)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
