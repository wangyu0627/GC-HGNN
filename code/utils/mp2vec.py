from tqdm import tqdm
import torch
import scipy.sparse as sp

def metapath2vec_train(args, model, epoch, device):
    for e in tqdm(range(epoch), desc="Metapath2vec Training"):
        loader = model.loader(batch_size=args.pre_batch_size, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.pre_lr)
        model.to(device)
        model.train()

        total_loss = 0
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    return model

def get_pos(target_emb, La_pos_num):
    # Location-aware positive samples
    f = target_emb
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    sim = torch.exp(torch.mm(f, f.t()) / torch.mm(f_norm, f_norm.t()))
    top_k_values1, top_k_indices1 = torch.topk(sim, La_pos_num, dim=1)
    La_pos = torch.zeros_like(sim)
    La_pos.scatter_(1, top_k_indices1, 1)
    La_pos = sp.coo_matrix(La_pos)

    return La_pos