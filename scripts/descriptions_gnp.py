from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd
from lassort import localAssortF
from matplotlib import pyplot as plt

N = 800


def run_null_analysis(s0):
    G = nx.gnp_random_graph(N, 0.075)
    E = nx.convert_matrix.to_pandas_edgelist(G).values

    # adjacency matrix
    A = nx.convert_matrix.to_scipy_sparse_matrix(G)
    
    # stub count is twice the edge count
    m2 = 2 * len(E)
    
    labels = np.hstack(
        [np.zeros(s0), 
         np.ones(N - s0)])
    
    _, assortT, _ = localAssortF(E,labels,pr=np.arange(0,1,0.1))
    
    # average score for first group
    T0 = assortT[:s0].mean()
    # average score for second group
    T1 = assortT[s0:].mean()

    # intra-community edge density for each group
    e0 = A[:s0, :s0].sum() / m2
    e1 = A[s0:, s0:].sum() / m2

    # degree proportion for each group
    a0 = A[:s0,:].sum() / m2
    a1 = A[s0:,:].sum() / m2

    # compute modularity
    sum_ag = a0**2 + a1**2
    Q = e0 + e1 - sum_ag
    Qmax = 1 - sum_ag
    
    # global assortativity
    r = Q/(1-Qmax)
    
    return s0, T0, T1, Qmax, Q, r


if __name__ == "__main__":
    # find the size of the smallest group
    s0s = np.arange(10, 401, 10)

    n_trials = 20
    s0s = pd.Series(np.hstack([s0s]*n_trials))
    
    results = s0s.apply(run_null_analysis)
    
    columns = ["s0", "T0", "T1", "Qmax", "Q", "r"]
    df = pd.DataFrame(
        results.to_list(),
        columns=columns
    )
    df["s0"] = s0s
    
    # save processed data just in case something goes wrong
    # in the plot
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M')
    df.to_csv(f"data/summary_erdos_renyi_{date_str}.csv", index=False)

    # save mirror image of data
    # e.g. group 1 larger than group 2
    # for nicer plot
    df_rev = df.copy()
    df_rev["s0"] = N - df_rev.s0

    # switch T0, T1
    # swiith a0, a1
    # switch e0, e1
    df_rev.columns = ["s0", "T1", "T0", "Qmax", "Q", "r"]
    df = pd.concat([df,df_rev], axis=0).sort_values("s0").reset_index(drop=True)
    
    grpd = df.groupby("s0")

    T0_mean = grpd.T0.mean()
    T1_mean = grpd.T1.mean()
    r_mean = grpd.r.mean()

    T0_std = grpd.T0.std()
    T1_std = grpd.T1.std()
    r_std = grpd.r.std()

    fig, ax = plt.subplots()
    ax.set_ylim([-0.5, 0.5])

    ax.errorbar(T0_mean.index, T0_mean.values, T0_std.values)
    ax.errorbar(T1_mean.index, T1_mean.values, T1_std.values)
    ax.errorbar(r_mean.index, r_mean.values, r_std.values)

    plt.legend([
        "Group 1 $r_{\ell}$",
        "Group 2 $r_{\ell}$",
        "$r_{global}$",
    ])

    plt.xlabel("Size of Group 1")
    plt.ylabel("Assortativity Score")
    plt.title("Local vs. Global Assortativity on Erdos Renyi Random Graph")

    plt.tight_layout()
    plt.savefig(f"images/erdos_renyi_{date_str}.png")
