#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:58:54 2020

@author: s1303034
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:11:05 2019

@author: rosalyn
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm, colors as mcolors
import sys

np.set_printoptions(threshold=sys.maxsize)

sns.set()

def matrix_plot_labels(df):
    explabels = [x[0] for x in df.index]
    datasetlabels = [x[1] for x in df.index]
    points = [x[2] for x in df.index]
    labels = datasetlabels
    unique_ds = []
    unique_ds.append([labels[0], points[0]])
    for x in range(len(labels) - 1):
        if labels[x + 1] != labels[x]:
            unique_ds.append([labels[x + 1], x + 1])
    ticklabels = [unique_ds[x][0] for x in range(len(unique_ds))]
    # Renaming ticklabels
    ticklabel_dict = {"NMCPD": "NMC",
                      "NMCPD_D": "NMC",
                      "SLACD": "SLAC",
                      "BCDMSD": "BCDMS",
                      "DYE886_D": "NuSea"}
    ticklabels = [ticklabel_dict[ticklabel] for ticklabel in ticklabels]
    startlocs = [unique_ds[x][1] for x in range(len(unique_ds))]
    startlocs += [len(labels)]
    ticklocs = [0 for x in range(len(startlocs) - 1)]
    for i in range(len(startlocs) - 1):
        ticklocs[i] = 0.5 * (startlocs[i + 1] + startlocs[i])
    return ticklocs, ticklabels, startlocs

def corrmat_plot(matrix, labeldf, descrip, label):
    fig, ax = plt.subplots(figsize=(7,7))
    diag_minus_half = (np.diagonal(matrix))**(-0.5)
    corrmat = diag_minus_half[:,np.newaxis]*matrix*diag_minus_half
    matrixplot = ax.matshow(corrmat,
                            cmap=cm.Spectral_r,
                            vmin = -1,
                            vmax = +1)
    cbar=fig.colorbar(matrixplot, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
  #  ax.set_title(f"{descrip} correlation matrix", fontsize=15)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(labeldf)
    plt.xticks(ticklocs, ticklabels, rotation=45, fontsize=15)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(ticklocs, ticklabels, fontsize=15)
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.vlines(startlocs_lines, 0, 1, transform=ax.get_xaxis_transform(), linestyles="dashed")
    ax.hlines(startlocs_lines,0, 1, transform=ax.get_yaxis_transform(), linestyles='dashed')
    ax.margins(x=0, y=0)
    plt.savefig(f"../../plots/covmats/covmats_{descrip}_{label}.png")
    return fig

def covmat_plots(label, fp1_table, fp2_table, fp1_covmat):

    expcovmat = fp1_covmat.values

    T_fp2 = fp2_table["theory_central"]
    D = fp1_table["data_central"]
    T_fp1 = fp1_table["theory_central"]
    
    # Calculating errors on T_fp1 by taking standard deviation
    T_fp1_reps = fp1_table.loc[:, fp1_table.columns.str.contains("rep")]
    nrep = len(T_fp1_reps.values.T)

    T_fp2_repeat = np.tile(T_fp2.values, (nrep,1)).T
    deltas = T_fp1_reps.values - T_fp2_repeat

    T_fp1_repeat = np.tile(T_fp1.values, (nrep,1)).T
    deltasprime =  T_fp1_reps.values - T_fp1_repeat

    covmat = (1/nrep) * deltas@deltas.T
    np.savetxt(f"covmatrix_{label}.csv", covmat, delimiter=",")
    covmatprime = (1/nrep) * deltasprime@deltasprime.T
    np.savetxt(f"covmatrix_shift_{label}.csv", covmatprime, delimiter=",")
    
    normcovmat = covmat/np.outer(D.values, D.values)
    normexpcovmat = expcovmat/np.outer(D.values, D.values)
    expsqrtdiags = np.sqrt(np.diag(normexpcovmat))
    totcovmat = covmat + expcovmat
    totnormcovmat = totcovmat/np.outer(D.values, D.values)
    
    fig_th = corrmat_plot(covmat, fp1_table, "theory", label)
    fig_exp = corrmat_plot(expcovmat, fp1_table, "experiment", label)
    fig_tot = corrmat_plot(totcovmat, fp1_table, "total", label)

    # Diag element plot
    sqrtdiags = np.sqrt(np.diag(normcovmat))
    totsqrtdiags = np.sqrt(np.diag(totnormcovmat))
    fig_diag, ax2 = plt.subplots(figsize=(20,7))
    ax2.plot(100*sqrtdiags, '-o', color="darkorange", label="S")
    ax2.plot(100*expsqrtdiags, '-o', color="purple", label="C")
    ax2.plot(100*totsqrtdiags, '-o', color="deepskyblue", label="C+S")
    ax2.set_ylabel("% of data", fontsize=20)
    ax2.set_title(f"Diagonal elements of covariance matrix",
                  fontsize=28)
    ax2.set_ylim([0, 35])
    ticklocs, ticklabels, startlocs = matrix_plot_labels(fp1_table)
    plt.xticks(ticklocs, ticklabels, rotation=30, ha="right", fontsize=15)
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    plt.yticks(fontsize=16)
    ax2.vlines(startlocs_lines, ymin, ymax, linestyles="dashed")
    ax2.margins(x=0, y=0)
    ax2.legend(fontsize=15)
    plt.savefig(f"../../plots/covmats/diag_covmat_{label}.png")

    return fig_th, fig_exp, fig_tot, fig_diag

# Loading observables

chorus = pd.read_table(
    "../observables/CHORUS_nuc/output/tables/group_dataset_inputs_by_experiment0_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

nutev = pd.read_table(
    "../observables/NUTEV_nuc/output/tables/group_dataset_inputs_by_experiment0_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

e605 = pd.read_table(
    "../observables/E605_nuc/output/tables/group_dataset_inputs_by_experiment0_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

chorus_proton = pd.read_table(
    "../observables/proton/output/tables/group_dataset_inputs_by_experiment0_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

nutev_proton = pd.read_table(
    "../observables/proton/output/tables/group_dataset_inputs_by_experiment1_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

e605_proton = pd.read_table(
    "../observables/proton/output/tables/group_dataset_inputs_by_experiment2_group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

proton =  pd.concat([chorus_proton,nutev_proton,e605_proton])
nuclear = pd.concat([chorus,nutev,e605])