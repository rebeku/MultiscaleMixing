from datetime import datetime
from lassort import load, localAssortF
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def assort_vs_freq(col_name, col_i, m):
    """
    Compute local assortativity vs. frequency for each class
    """
    start = datetime.now()
    E, M = load(
        networkfile, 
        metadatafile, 
        zero_index=1, 
        meta_col=col_i,
        header=True,
        reindex=True,
        missing_value=0
    )

    _, assortT, _ = localAssortF(E,M,pr=np.arange(0,1,0.1))
    end = datetime.now()
    print(f"Ran multiscale mixing for {col_name} in {(end-start).total_seconds()} seconds.")
    
    grps = pd.DataFrame([m[col_name].values, assortT]).T
    grps.columns = [col_name, "T"]

    assort = grps.groupby(col_name)[["T"]].mean()
    assort["n"] = grps[col_name].value_counts()
    return assort.drop(0)


if __name__ == "__main__":
    # load Wellesley network from the Facebook 100
    networkfile = '../../problem_sets/2/facebook100txt/Wellesley22.txt'
    metadatafile = '../../problem_sets/2/facebook100txt/Wellesley22_attr.txt'

    m = pd.read_csv(metadatafile,sep="\t")
    
    dorms = assort_vs_freq("dorm", 4, m)
    dorms.plot(kind="scatter", x="n", y="T")
    
    plt.xlabel("Class Size")
    plt.ylabel("Average $r_{\ell}$")
    plt.title("Assortativity by dorm at Wellesley")
    plt.savefig("images/wellesley_dorms.png")

    majors = assort_vs_freq("major", 3, m)
    majors.plot(kind="scatter", x="n", y="T")

    plt.xlabel("Class Size")
    plt.ylabel("Average $r_{\ell}$")
    plt.title("Assortativity by major at Wellesley")
    plt.savefig("images/wellesley_major.png")
