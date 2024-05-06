import numpy
import torch
from utils import load_data, set_params, evaluate
from utils.load_data import TensorToDGL, preprocess_features, sparse_mx_to_torch_sparse_tensor
from module.HGCL import HGCL
import warnings
import datetime
import pickle as pkl
import os
import random
from tqdm import tqdm
from torch_geometric.nn.models import MetaPath2Vec
from utils.mp2vec import metapath2vec_train, get_pos

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train(i):
    data_path = '../data/'
    (nei_index, predict, meta_paths, mps, feats, pos, label, idx_train, idx_val, idx_test), g, processed_metapaths = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    # feats_dim_list = [i.shape[1] for i in feats]
    G_file = open(os.path.join(data_path + args.dataset, args.dataset + "_hg.pkl"), "rb")
    G = pkl.load(G_file)
    G = G.to(device)
    G_file.close()
    gs = [g.to(device) for g in mps]
    gs = TensorToDGL(gs)
    P = len(meta_paths[predict])
    print("seed ",args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", P)

    # metapath2vec for pretrain and pos
    pretrain_model = MetaPath2Vec(g.edge_index_dict,
                                  args.pre_embedding_dim,
                                  processed_metapaths,
                                  args.pre_walk_length,
                                  args.pre_context_size,
                                  args.pre_walks_per_node,
                                  args.pre_num_negative_samples,
                                  sparse=True
                                  )
    metapath2vec_train(args, pretrain_model, args.pre_epoch, device)

    target_feat = pretrain_model('target').detach()
    # free up memory
    del pretrain_model
    target_feat = target_feat.cpu()
    torch.cuda.empty_cache()
    target_feat = torch.FloatTensor(preprocess_features(target_feat))

    # update features and pos
    La_pos = get_pos(target_feat, args.La_pos_num)
    La_pos = sparse_mx_to_torch_sparse_tensor(La_pos)
    pos = [La_pos, pos]
    features = {ntype: feats[i].cuda() for i, ntype in enumerate(G.ntypes)}
    target_feat = torch.hstack([features[predict], target_feat.cuda()])
    features[predict] = target_feat.cuda()


    model = HGCL(G, gs, args.fanout, args.edge_dir, P, args.ntypes, args.hidden_dim, args.out_dim, args.encoder,
                 args.decoder, args.num_heads, args.dropout, args.feat_drop, args.attn_drop, args.enc_num_layer,
                 args.dec_num_layer, args.mask_rate,args.remask_rate, args.num_remasking, args.alpha,
                 args.beta, args.tau, args.lam).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        # features = [feat.cuda() for feat in features]
        pos = [i.cuda() for i in pos]
        # pos = pos.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0

    starttime = datetime.datetime.now()
    # for epoch in range(args.nb_epochs):
    for epoch in tqdm(range(args.nb_epochs), desc='Training Epochs'):
        model.train()
        optimiser.zero_grad()
        loss = model(features, pos, predict)
        # print("loss ", loss.data.cpu())
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'HGCL_'+own_str+'.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            # print('Early stopping!')
            break
        loss.backward()
        optimiser.step()
        
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('HGCL_'+own_str+'.pkl'))
    model.eval()
    os.remove('HGCL_'+own_str+'.pkl')
    embeds = model.get_embeds(features, predict)
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")
    
    # if args.save_emb:
    # f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
    # pkl.dump(embeds.cpu().data.numpy(), f)
    # f.close()


if __name__ == '__main__':
    train(i=0)
