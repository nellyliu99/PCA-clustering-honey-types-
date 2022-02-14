
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:04:49 2021

@author: Nelly
"""

#utility
import numpy as np
from numpy import trapz
from scipy import linalg
from scipy.stats import linregress
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from tkinter import filedialog
from tkinter import Tk
import os
from collections import OrderedDict
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,fbeta_score
import seaborn as sn
from operator import itemgetter

from sys import exit
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"


#sklearn
from sklearn.decomposition import PCA


#Plot

def confidence_ellipse(x, y, ax, n_std=2, facecolor='None', linestyle='-', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor, ls=linestyle, alpha=1,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

            
def plot_PCA(data, filename):

    X = StandardScaler().fit_transform(data.drop(['label'], axis=1).values)
    y = data['label'].values

    colors = ['r', 'g', 'b', 'red', 'magenta', 'orange', 'brown']
    markers_code = ['o', '>', '^', 's', 'D']
    lw = 2


    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    pcs = []
    for i in np.unique(y):
        pcs.append([np.array([i[0] for i in X_pca[y == i]]), np.array([i[1] for i in X_pca[y == i]])])

    fig, ax = plt.subplots()
    for color, target_name, markers, pc_data in zip(colors, np.unique(y), markers_code, pcs):
        plt.scatter(X_pca[y == target_name, 0], X_pca[y == target_name, 1], color=color, alpha=1, lw=lw,
                    marker=markers,
                    label=target_name, edgecolor=color, linewidth=1)
        confidence_ellipse(pc_data[0], pc_data[1], ax, edgecolor=color, linestyle='--')

    total_var = pca.explained_variance_ratio_.sum() * 100
    plt.xlabel("PC1: " + str(round(pca.explained_variance_ratio_[0] * 100, 2)) + "%", fontsize=18)
    plt.ylabel("PC2: " + str(round(pca.explained_variance_ratio_[1] * 100, 2)) + "%", fontsize=18)
    plt.legend(shadow=False, scatterpoints=1, fontsize=12).get_frame().set_edgecolor('k')
    plt.title( filename + f'({total_var:.2f}%)', fontsize=22)
    plt.show()
    fig.savefig('%s.png'%filename, dpi=1000, bbox_inches='tight')
    

            
root = Tk()
root.attributes("-topmost", True)
alamat = filedialog.askdirectory(initialdir=os.path.dirname(os.path.realpath(__file__)),title="select the folder to be extracted")
root.withdraw()
for file in os.listdir(alamat):
        if file.endswith(".csv"):
            list_file = [file]
        for file in list_file:
            if file.split("_")[-2] == 'W1':
                w = "1W"
            elif file.split("_")[-2] == 'W2':
                w = "2W"
            elif file.split("_")[-2] == 'W3':
                w = "3W"
            elif file.split("_")[-2] == 'W4':
                w = "4W"
            else:
                w = file.split("_")[-2]
                
            print('plotting %s'%file + '...')
            data = pd.read_csv(file)
            plot_PCA(data, filename='EN-%s-PCA'%w)

     
            
