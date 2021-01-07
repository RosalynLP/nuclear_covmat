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

sns.set_style("white")

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
    ticklabel_dict = {"CHORUSNUPb": r"CHORUS $\nu$",
                      "CHORUSNBPb": r"CHORUS $\bar{\nu}$",
                      "NTVNUDMNFe": r"NuTeV $\nu$",
                      "NTVNBDMNFe": r"NuTeV $\bar{\nu}$",
                      "DYE605": "DYE605"}
    ticklabels = [ticklabel_dict[ticklabel] for ticklabel in ticklabels]
    startlocs = [unique_ds[x][1] for x in range(len(unique_ds))]
    startlocs += [len(labels)]
    ticklocs = [0 for x in range(len(startlocs) - 1)]
    for i in range(len(startlocs) - 1):
        ticklocs[i] = 0.5 * (startlocs[i + 1] + startlocs[i])
    return ticklocs, ticklabels, startlocs

def corrmat_plot(matrix, labeldf, descrip, label):
    fig, ax = plt.subplots(figsize=(10,10))
    diag_minus_half = (np.diagonal(matrix))**(-0.5)
    corrmat = diag_minus_half[:,np.newaxis]*matrix*diag_minus_half
    matrixplot = ax.matshow(np.nan_to_num(corrmat),
                            cmap=cm.Spectral_r,
                            vmin = -1,
                            vmax = +1)
    cbar=fig.colorbar(matrixplot, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(f"{descrip} correlation matrix", fontsize=25)
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

def diag_element_plot(normcovmat, totnormcovmat, expsqrtdiags, descrip, label):
    # Diag element plot
    sqrtdiags = np.sqrt(np.diag(normcovmat))
    totsqrtdiags = np.sqrt(np.diag(totnormcovmat))
    fig_diag, ax2 = plt.subplots(figsize=(20,7))
    ax2.plot(100*sqrtdiags, 'o', color="darkorange", label="S")
    ax2.plot(100*expsqrtdiags, 'o', color="purple", label="C")
    ax2.plot(100*totsqrtdiags, 'o', color="deepskyblue", label="C+S")
    ax2.set_ylabel("% of data", fontsize=20)
    ax2.set_title(f"Diagonal elements of covariance matrix, {descrip}",
                  fontsize=28)
    ax2.set_ylim([0, 120])
    ticklocs, ticklabels, startlocs = matrix_plot_labels(proton)
    plt.xticks(ticklocs, ticklabels, rotation=30, ha="right", fontsize=15)
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ymin, ymax = ax2.get_ylim()
    xmin, xmax = ax2.get_xlim()
    plt.yticks(fontsize=16)
    ax2.vlines(startlocs_lines, ymin, ymax, linestyles="dashed")
    ax2.margins(x=0, y=0)
    ax2.legend(fontsize=15)
    plt.savefig(f"../../plots/covmats/diag_covmat_{descrip}.png")
    return fig_diag

def make_covmat(label, proton, nuclear, expcov):
    
    expcovmat = expcov.values

    proton_theory = proton["theory_central"]
    D = nuclear["data_central"]
    nuclear_theory = nuclear["theory_central"]
    
    
    # Calculating errors on T_fp1 by taking standard deviation
    proton_reps = proton.loc[:, proton.columns.str.contains("rep")]
    nuclear_reps = nuclear.loc[:, nuclear.columns.str.contains("rep")]
    nprotonrep = len(proton_reps.values.T)
    nnuclearrep = len(nuclear_reps.values.T)

    proton_repeat = np.tile(proton_theory.values, (nnuclearrep,1)).T
    deltas = nuclear_reps.values - proton_repeat

    nuclear_repeat = np.tile(nuclear_theory.values, (nnuclearrep,1)).T
    deltasprime =  nuclear_reps.values - nuclear_repeat

    shift = nuclear_theory - proton_theory
    err = np.std(nuclear_reps - proton_repeat, axis=1)
  
    fig, ax = plt.subplots(figsize=(20,7))
    ax.errorbar(np.arange(len(shift)), 100*(shift/D).values, yerr=100*(err/D).values, color="k")
    #ax.plot(100*((shift-err)/D).values, color="k", alpha=0.5)
   # ax.plot(100*((shift+err)/D).values, color="k", alpha=0.5)
    ax.set_ylabel("% of data", fontsize=20)
    ax.set_title("Shift", fontsize=25)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(proton)
    plt.xticks(ticklocs, ticklabels, rotation=30, ha="right", fontsize=15)
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    plt.yticks(fontsize=16)
    ax.vlines(startlocs_lines, ymin, ymax, linestyles="dashed")
    ax.margins(x=0, y=0)
    ax.legend(fontsize=15)
    plt.savefig(f"../../plots/covmats/shift.png")
    
    covmat = (1/nnuclearrep) * deltas@deltas.T
    covmat_df = pd.DataFrame(covmat, index=proton.index, columns=proton.index)
    covmat_df.to_csv(f"covmatrix_{label}.csv")
    covmatprime = (1/nnuclearrep) * deltasprime@deltasprime.T
    covmatprime_df = pd.DataFrame(covmatprime, index=proton.index, columns=proton.index)
    covmatprime_df.to_csv(f"covmatrix_shift_{label}.csv")
    
    normcovmat = covmat/np.outer(D.values, D.values)
    normcovmatshift = covmatprime/np.outer(D.values, D.values)
    normexpcovmat = expcovmat/np.outer(D.values, D.values)
    expsqrtdiags = np.sqrt(np.diag(normexpcovmat))
    totcovmat = covmat + expcovmat
    totcovmatshift = covmatprime + expcovmat
    totnormcovmat = totcovmat/np.outer(D.values, D.values)
    totnormcovmatshift = totcovmatshift/np.outer(D.values, D.values)
    
    fig_th = corrmat_plot(covmat, proton, "Theory", label)
    fig_exp = corrmat_plot(expcovmat, proton, "Experiment", label)
    fig_tot = corrmat_plot(totcovmat, proton, "Total", label)
    fig_shiftth = corrmat_plot(covmatprime, proton, "Theory, shifted", label)
    fig_shifttot = corrmat_plot(totcovmatshift, proton, "Total, shifted", label)
    
    diag_element_plot(normcovmat, totnormcovmat, expsqrtdiags, "deweighted", label)
    diag_element_plot(normcovmatshift, totnormcovmatshift, expsqrtdiags, "shifted", label)
    
    return fig_th, fig_exp, fig_tot, fig_shiftth, fig_shifttot

# Loading observables

chorus = pd.read_table(
    "../observables/CHORUS_nuc/output/tables/group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

nutev = pd.read_table(
    "../observables/NUTEV_nuc/output/tables/group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

e605 = pd.read_table(
    "../observables/E605_nuc/output/tables/group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

proton = pd.read_table(
    "../observables/proton/output/tables/group_result_table.csv",
    dtype={"user_id": float},
    index_col=[0,1,2]
)

expcov = pd.read_table(
    "../observables/proton/output/tables/groups_covmat.csv",
    dtype={"user_id": float},
    index_col=[0,1,2], header=[0,1,2]
)

nuclear = pd.concat([chorus,nutev,e605])

make_covmat("nuclear", proton, nuclear, expcov)