import torch
import scipy.sparse as sp

def acm():
    # Location-aware positive samples
    f_pap = torch.load('emb/pap.npz')
    f_psp = torch.load('emb/psp.npz')
    f = torch.cat((f_pap, f_psp), dim=1).cpu()
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    sim = torch.exp(torch.mm(f, f.t()) / torch.mm(f_norm, f_norm.t()))
    top_k_values1, top_k_indices1 = torch.topk(sim, 1, dim=1)
    La_pos = torch.zeros_like(sim)
    La_pos.scatter_(1, top_k_indices1, 1)
    La_pos = sp.coo_matrix(La_pos)
    sp.save_npz('acm/la_pos.npz', La_pos)
    print(La_pos.sum(-1).max(), La_pos.sum(-1).min(), La_pos.sum(-1).mean())


    # Semantic-aware positive samples
    pap = sp.load_npz("./acm/pap.npz")
    pap = pap / pap.sum(axis=-1).reshape(-1, 1)
    psp = sp.load_npz("./acm/psp.npz")
    psp = psp / psp.sum(axis=-1).reshape(-1, 1)
    all =  (pap + psp).A.astype("float32")
    all = torch.tensor(all)
    top_k_values2, top_k_indices2 = torch.topk(all, 5, dim=1)
    Sa_pos = torch.zeros_like(sim)
    Sa_pos.scatter_(1, top_k_indices2, 1)
    Sa_pos = sp.coo_matrix(Sa_pos)
    sp.save_npz('acm/sa_pos.npz', Sa_pos)
    print(Sa_pos.sum(-1).max(), Sa_pos.sum(-1).min(), Sa_pos.sum(-1).mean())

def freebase():
    # Location-aware positive samples
    f_mam = torch.load('emb/freebase/mam.npz')
    f_mdm = torch.load('emb/freebase/mdm.npz')
    f_mwm = torch.load('emb/freebase/mwm.npz')
    f = torch.cat((f_mam, f_mdm, f_mwm), dim=1).cpu()
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    sim = torch.exp(torch.mm(f, f.t()) / torch.mm(f_norm, f_norm.t()))
    top_k_values1, top_k_indices1 = torch.topk(sim, 5, dim=1)
    La_pos = torch.zeros_like(sim)
    La_pos.scatter_(1, top_k_indices1, 1)
    La_pos = sp.coo_matrix(La_pos)
    sp.save_npz('freebase/la_pos.npz', La_pos)
    print(La_pos.sum(-1).max(), La_pos.sum(-1).min(), La_pos.sum(-1).mean())


    # Semantic-aware positive samples
    mam = sp.load_npz("./freebase/mam.npz")
    mam = mam / mam.sum(axis=-1).reshape(-1, 1)
    mdm = sp.load_npz("./freebase/mdm.npz")
    mdm = mdm / mdm.sum(axis=-1).reshape(-1, 1)
    mwm = sp.load_npz("./freebase/mwm.npz")
    mwm = mwm / mwm.sum(axis=-1).reshape(-1, 1)
    all =  (mam + mdm + mwm).A.astype("float32")
    all = torch.tensor(all)
    top_k_values2, top_k_indices2 = torch.topk(all, 80, dim=1)
    Sa_pos = torch.zeros_like(sim)
    Sa_pos.scatter_(1, top_k_indices2, 1)
    Sa_pos = sp.coo_matrix(Sa_pos)
    sp.save_npz('freebase/sa_pos.npz', Sa_pos)
    print(Sa_pos.sum(-1).max(), Sa_pos.sum(-1).min(), Sa_pos.sum(-1).mean())

def dblp():
    # Location-aware positive samples
    f_apa = torch.load('emb/dblp/apa.npz')
    f_apcpa = torch.load('emb/dblp/apcpa.npz')
    f_aptpa = torch.load('emb/dblp/aptpa.npz')
    f = torch.cat((f_apa, f_apcpa, f_aptpa), dim=1).cpu()
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    sim = torch.exp(torch.mm(f, f.t()) / torch.mm(f_norm, f_norm.t()))
    top_k_values1, top_k_indices1 = torch.topk(sim, 50, dim=1)
    La_pos = torch.zeros_like(sim)
    La_pos.scatter_(1, top_k_indices1, 1)
    La_pos = sp.coo_matrix(La_pos)
    sp.save_npz('dblp/la_pos.npz', La_pos)
    print(La_pos.sum(-1).max(), La_pos.sum(-1).min(), La_pos.sum(-1).mean())


    # Semantic-aware positive samples
    apa = sp.load_npz("./dblp/apa.npz")
    apa = apa / apa.sum(axis=-1).reshape(-1, 1)
    apcpa = sp.load_npz("./dblp/apcpa.npz")
    apcpa = apcpa / apcpa.sum(axis=-1).reshape(-1, 1)
    aptpa = sp.load_npz("./dblp/aptpa.npz")
    aptpa = aptpa / aptpa.sum(axis=-1).reshape(-1, 1)
    all =  (apa + apcpa + aptpa).A.astype("float32")
    all = torch.tensor(all)
    top_k_values2, top_k_indices2 = torch.topk(all, 1000, dim=1)
    Sa_pos = torch.zeros_like(sim)
    Sa_pos.scatter_(1, top_k_indices2, 1)
    Sa_pos = sp.coo_matrix(Sa_pos)
    sp.save_npz('dblp/sa_pos.npz', Sa_pos)
    print(Sa_pos.sum(-1).max(), Sa_pos.sum(-1).min(), Sa_pos.sum(-1).mean())

def aminer():
    # Location-aware positive samples
    f_pap = torch.load('emb/aminer/pap.npz')
    f_prp = torch.load('emb/aminer/prp.npz')
    f = torch.cat((f_pap, f_prp), dim=1).cpu()
    f_norm = torch.norm(f, dim=-1, keepdim=True)
    sim = torch.exp(torch.mm(f, f.t()) / torch.mm(f_norm, f_norm.t()))
    top_k_values1, top_k_indices1 = torch.topk(sim, 2, dim=1)
    La_pos = torch.zeros_like(sim)
    La_pos.scatter_(1, top_k_indices1, 1)
    La_pos = sp.coo_matrix(La_pos)
    sp.save_npz('aminer/la_pos.npz', La_pos)
    print(La_pos.sum(-1).max(), La_pos.sum(-1).min(), La_pos.sum(-1).mean())


    # Semantic-aware positive samples
    pap = sp.load_npz("./aminer/pap.npz")
    pap = pap / pap.sum(axis=-1).reshape(-1, 1)
    prp = sp.load_npz("./aminer/prp.npz")
    prp = prp / prp.sum(axis=-1).reshape(-1, 1)
    all =  (pap + prp).A.astype("float32")
    all = torch.tensor(all)
    top_k_values2, top_k_indices2 = torch.topk(all, 15, dim=1)
    Sa_pos = torch.zeros_like(sim)
    Sa_pos.scatter_(1, top_k_indices2, 1)
    Sa_pos = sp.coo_matrix(Sa_pos)
    sp.save_npz('aminer/sa_pos.npz', Sa_pos)
    print(Sa_pos.sum(-1).max(), Sa_pos.sum(-1).min(), Sa_pos.sum(-1).mean())


if __name__ == '__main__':
    acm()
  # freebase()
  # dblp()
  # aminer()