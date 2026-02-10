import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    p=np.array(p)
    q=np.array(q)
    final_q = q+eps
    final_p = p+eps
    kld = np.sum(final_p* np.log(final_p/final_q))
    return kld