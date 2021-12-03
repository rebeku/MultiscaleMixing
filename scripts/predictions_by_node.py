from matplotlib import pyplot as plt
from lassort import localAssortF
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime


def truncated_power_law(gamma, k_0, n, rng, size=None):
    """
    Generate a sample of size *size* from a power law distribution
    with mininmum *k_0*, maximum *n*, and power *gamma*
    """
    k_0=np.float64(k_0)
    gamma=np.float64(gamma)
    n=np.float64(n)
    
    if size:
        U = rng.uniform(size=size)
    else:
        U = rng.uniform()
        
    return (
        (k_0**(1-gamma) - 
             ( k_0**(1-gamma) - n**(1-gamma) ) * U 
        )**(1/(1-gamma))
    )


def proportional_partition(props, N):
    sizes = (N * props).astype(int)
    
    # adjust for any count lost in rounding
    diff = N - sizes.sum()
    for i in range(diff):
        sizes[i] += 1

    return sizes


def predict_msm_block(B, Z, pr, N, C):
    """
    predict multiscale mixing based on a DC-SBM
    mixing matrix
    
    Returns
    -------
    
    predM : array_like 
        C x len(pr) array of expected local assortativities for each block.
        predM[C, i] corresponds to the expected local assortativity of nodes
        in cluster C with alpha = i
    
    predT : array_like
        C x 1 array of expected multiscale assortattivities for each block.
        predT[C] is the expected multiscale assortativity for community C
    """
    D = np.diag(B.sum(axis=0))
    P = np.linalg.inv(D) @ B
    
    ar2 = (B.sum(axis=0) / B.sum())**2
    qmax = 1 - ar2.sum()
    
    predM = np.zeros((C, len(pr)))
    predT = np.zeros((C,1))

    for i in range(C):
        u = np.where(Z[:,i]==1)[0][0]
        
        for j, alpha in enumerate((1-pr)):
            pi = np.zeros(N)
            pi[u] = 1
            
            alpha_pi_t = pi @ Z * alpha
            inv_transform = np.linalg.inv(np.eye(C) - (1-alpha)*P)
            
            # p is blockwise PPR
            p = alpha_pi_t @ inv_transform
            
            # compute multiscale mixing from PPR
            eggl = p * np.diag(B)/ B.sum(axis=0)
            predM[i,j] = (eggl.sum() - ar2.sum()) / qmax
            
            
    # use the trapezoid rule to estimate predT
    delta = (pr - np.roll(pr,1))[1:]

    traps = (predM + np.roll(predM, 1))[:,1:] / 2
    predT = (traps * delta).sum(axis=1) / pr.max()
    return predM, predT
            
    
def generate_dc_sbm(B, theta, Z, N, rng):
    """
    generate DC-SBM for use in actual multiscale mixing model
    """
    # generate random graph
    A = np.zeros((N,N), dtype=int)
    choices = rng.uniform(size=N*(N-1)//2)

    # probabilities for an edge between each pair of nodes
    theta = np.diag(theta)
    p = (theta @ Z @ B @ Z.T @ theta)
    assert p.max() < 1, p.max()

    # this sucks but yeet
    A = np.zeros((N,N), dtype=int)

    k = 0
    for i in range(N):
        for j in range(i+1, N):
            if choices[k] < p[i,j]:
                A[i,j] = 1
                A[j,i] = 1
            k+=1
    
    G = nx.convert_matrix.from_numpy_matrix(A)
    E = nx.convert_matrix.to_pandas_edgelist(G).values[:,:2]
    M = np.where(Z)[1]
    
    return E, M


def run_simulation(
    N, 
    C, 
    props, 
    n_trials, 
    pr_in, 
    pr_out, 
    rng
):
    """
    Run the simulation
    """

    # Z is block membership matrix
    Z = np.zeros((N, C), dtype=int)

    sizes = proportional_partition(props, N)
    sizes.sort()

    thresholds = np.zeros(len(sizes) + 1, dtype=int)
    thresholds[1:] = sizes

    for i in range(2, len(thresholds)):
        thresholds[i:] += sizes[i-2]

    for i in range(C):
        Z[thresholds[i]:thresholds[i+1]][:,i] = 1
        
    # mixing matrix
    B = np.ones((C,C)) * 8


    for i in range(C):
        B[i,i] = pr_in * sizes[i] * (sizes[i] - 1) / 2
        
        for j in range(i+1, C):
            # 5% of possible external edges exist
            B[i,j] = pr_out * sizes[i] * sizes[j]
            B[j,i] = B[i,j]
            
    # degree distribution
    # assign edge wealth to each node
    theta = []

    for size in sizes:
        s = rng.uniform(size=size)
        theta.append(s/s.sum())
        
    theta = np.hstack(theta)

    # predict values
    pr = np.arange(0,1,0.05)
    _, predT = predict_msm_block(B, Z, pr, N, C)

    # generate graph
    # this can fail if a probability is > 1
    E = None
    M = None

    for k in range(5):
        try:
            E, M = generate_dc_sbm(B, theta, Z, N, rng)
            break
        except:
            continue
    
    # calculate local assortativity
    _, assortT, _ = localAssortF(E,M)

    scores = []
    for i in range(C):
        com = assortT[thresholds[i]:thresholds[i+1]]
        msm = com[np.where(~np.isnan(com))].mean()
        scores.append(msm) 
    
    df = pd.DataFrame(assortT.T, columns=["assortT"])
    df["block"] = pd.Series(df.index).apply(lambda x: np.where(thresholds <= x)[0][-1])

    # plot the distribution of local assortativity
    # within each block vs. block size
    means = df.groupby("block").assortT.mean()
    stds = df.groupby("block").assortT.std()
    cnts = df.block.value_counts().sort_index()

    plt.figure()
    plt.scatter(cnts, predT, color="tab:orange",marker="X")
    plt.errorbar(cnts, means, stds,fmt="P")
    plt.legend(["Predicted", "Observed"])
    plt.xlabel("Class Size")
    plt.ylabel("$r(\ell)$")

    if pr_in > pr_out:
        title = f"Assortative: N={N}"
        fname = f"images/assortative_N_{N}.png"
    else:
        title = f"Disassortative: N={N}"
        fname = f"images/disassortative_N_{N}.png"

    plt.title(title)
    plt.savefig(fname)
    print(f"Saved figure {title}")

    # plot the distribution of local assortativiity scores
    # within a single block
    print("Plotting block 9")
    plt.figure()
    df[df.block==9].assortT.plot(kind='hist')
    plt.xlabel("$r_\ell$")
    plt.title(f"Block 9 $r_\ell$: N={N}")

    # plot predicted value
    ymax = plt.gca().get_ylim()[1]
    plt.plot([predT[9], predT[9]], [0, ymax], "--")
    plt.legend(["Predicted", "Observed"])

    fname = fname[:-4] + "block_9.png"
    plt.savefig(fname)


if __name__=="__main__":
    Ns = [200,500,1000,5000]
    n_trials = 50
    C = 10 # number of blocks
    gamma = 2.2 # coefficient for modified power law
    k_0 = 8 # minimum expected block size
    
    pr_big = 0.16 # probability of edge between same-group nodes
    pr_small = 0.05 # probability of edge between out-group nodes

    rng = np.random.default_rng(11235811)
    
    # compute the proportion on nodes in each community
    # keep this fixed across network sizes
    props = truncated_power_law(gamma,k_0,(Ns[0])/2,rng,size=C)
    props = props / props.sum()
    props.sort()

    for N in Ns: 
        start = datetime.now()
        run_simulation(N, C, props, n_trials, pr_big, pr_small, rng)
        seconds = (datetime.now() - start).total_seconds()
        print(f"Ran simulation for N={N}, n_trials={n_trials} in {seconds} seconds.")
        
        start = datetime.now()
        run_simulation(N, C, props, n_trials, pr_small, pr_big, rng)
        seconds = (datetime.now() - start).total_seconds()
        print(f"Ran simulation for N={N}, n_trials={n_trials} in {seconds} seconds.")

    