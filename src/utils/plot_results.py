import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib
def plot_3D(X, Y, Z, C, labelx, labely, labelz, labelc, color_map, name):
    size = 50
    norm = Normalize()
    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.set_dpi(128)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter3D(X, Y, Z, facecolors=color_map(norm(C)), s=size, edgecolor='black')
    ax1.set_xlabel(labelx, fontweight="bold", fontsize=16, labelpad=20)
    ax1.set_ylabel(labely, fontweight="bold", fontsize=16, labelpad=20)
    ax1.set_zlabel(labelz, fontweight="bold", fontsize=16, labelpad=10)
    m1 = cm.ScalarMappable(cmap=color_map)
    m1.set_array(C)
    cbar = fig.colorbar(m1, shrink=0.5, aspect=12)
    cbar.set_label(labelc, fontsize=14, fontweight="bold")
    plt.savefig(name + '.png')


def plot_2D(X, Y, C, labelx, labely, labelc, color_map, name, tickss=None, shrink=1, aspect=12):
    size = 40
    norm = Normalize()
    fig = plt.figure()
    plt.style.use('seaborn-dark')
    fig.set_figheight(6)
    fig.set_figwidth(10)
    fig.set_dpi(96)
    plt.grid(color='gray', linestyle='--', linewidth=0.8)
    # plt.scatter(X, Y, facecolors=color_map(norm(C)), s=size, edgecolor='black', )
    plt.scatter(X, Y, c='blue', s=size, edgecolor='black')
    plt.xlabel(labelx, fontweight="bold", fontsize=20)
    plt.ylabel(labely, fontweight="bold", fontsize=20)
    # m = cm.ScalarMappable(cmap=color_map)
    # m.set_array(C)
    # cbar = fig.colorbar(m, shrink=shrink, aspect=aspect, ticks=tickss)
    # cbar.set_label(labelc, fontsize=14, fontweight="bold")
    plt.savefig(name + '.png')

def plot_norm_data(array, name, factor, xlim, maxim):
    array.sort()
    hmean = np.mean(array)
    hstd = np.std(array)
    median, q1, q3 = np.percentile(array, 50), np.percentile(array, 25), np.percentile(array, 75)
    sigma = hstd
    mu = hmean
    iqr = 3 * (q3 - q1)
    x1 = np.linspace(q1 - iqr, q1)
    x2 = np.linspace(q3, q3 + iqr)
    pdf1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
    pdf2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x2 - mu)**2 / (2 * sigma**2))
    pdf = stats.norm.pdf(array, hmean, hstd)
    arran = np.linspace(0, maxim, num=(70))
    plt.figure(figsize=(10, 6), dpi=96)
    plt.grid(color='gray', linestyle='--', linewidth=0.8)
    plt.plot(array, pdf * factor, label=f'Mean: {hmean:0.3f}\nStd: {hstd:0.3f}\nQ1: {q1:0.3f}\nQ3: {q3:0.3f}', linewidth=3)
    n, bins, patches = plt.hist(array, bins=arran, edgecolor='black', density=False)
    plt.fill_between(x1, pdf1 * factor, 0, alpha=.4, color='green')
    plt.fill_between(x2, pdf2 * factor, 0, alpha=.4, color='green')
    plt.xlabel(name, fontweight="bold", fontsize=20)
    plt.ylabel('No. Images', fontweight="bold", fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.xlim(xlim)
    plt.savefig(name + '.png')

def k_means():
    path = 'csv/kaggle_new_label.csv'
    csv1 = pd.read_csv(path)
    csv1 = csv1.sample(frac=1).reset_index(drop=True)
    SNR = csv1['SNR'].to_numpy()
    BLUR = csv1['Blur'].to_numpy()
    CONTRAST = csv1['Contrast'].to_numpy()
    BRISQUE = csv1['Brisque'].to_numpy()
    BRIGHT = csv1['Brightness'].to_numpy()
    LABEL = csv1['Qlabel'].to_numpy()
    # DR = csv1['DR'].to_numpy()
    size = 50
    array = np.array([SNR, BLUR, CONTRAST, BRISQUE, BRIGHT])
    data = array.transpose((1, 0))
    kmeans = KMeans(n_clusters=2).fit(data)
    centroids = kmeans.cluster_centers_
    print('Centroids:')
    print(f'[CLASS 0] -> SNR: {centroids[0][0]:0.3f}, BLUR: {centroids[0][1]:0.3f}, CONTRAST: {centroids[0][2]:0.3f}, BRISQUE: {centroids[0][3]:0.3f}, BRIGHTNESS: {centroids[0][4]:0.3f}')
    print(f'[CLASS 1] -> SNR: {centroids[1][0]:0.3f}, BLUR: {centroids[1][1]:0.3f}, CONTRAST: {centroids[1][2]:0.3f}, BRISQUE: {centroids[1][3]:0.3f}, BRIGHTNESS: {centroids[1][4]:0.3f}')
    labels_kmeans = kmeans.predict(data)
    # labels_kmeans = np.logical_not(labels_kmeans).astype(int)
    output_kmeans = labels_kmeans.tolist()
    # Getting the cluster centers
    Cluster = kmeans.cluster_centers_
    colores = ['red', 'green']
    asignar = []
    for row in labels_kmeans:
        asignar.append(colores[row])

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.set_dpi(96)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=asignar, s=size, edgecolor='black')
    ax1.set_xlabel('SNR', fontweight="bold", fontsize=14)
    ax1.set_ylabel('BLUR', fontweight="bold", fontsize=14)
    ax1.set_zlabel('CONTRAST', fontweight="bold", fontsize=14)
    plt.savefig('images/kmeans_cluster.png')

    l2 = labels_kmeans
    acc = np.sum(labels_kmeans == LABEL) / labels_kmeans.shape[0]
    print(acc)
    print(kappa(labels_kmeans, LABEL))

    cf_matrix = confusion_matrix(LABEL, LABEL)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='binary')
    ax.set_xlabel('CNN labels', fontsize=14, fontweight="bold")
    ax.set_ylabel('CNN labels', fontsize=14, fontweight="bold")
    ax.xaxis.set_ticklabels(['Low Quality', 'High Quality'])
    ax.yaxis.set_ticklabels(['Low Quality', 'High Quality'])
    plt.savefig('images/ideal_cfmatrix.png')

    cf_matrix = confusion_matrix(LABEL, labels_kmeans)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='binary')
    ax.set_xlabel('KMeans Group', fontsize=20, fontweight="bold")
    ax.set_ylabel('CNN labels', fontsize=20, fontweight="bold")
    ax.yaxis.set_ticklabels(['Low Quality', 'High Quality'], fontsize=14)
    ax.yaxis.set_ticklabels(['Low Quality', 'High Quality'], fontsize=14)
    plt.savefig('images/kmeans_cnn_cfmatrix.png')
    return labels_kmeans

def multilabel_cfmatrix():
    path = 'csv/kaggle_new_label.csv'
    csv1 = pd.read_csv(path)
    csv1 = csv1.sample(frac=1).reset_index(drop=True)
    LABEL = csv1['Qlabel'].to_numpy()
    DR = csv1['DR'].to_numpy()

    y_pred = []
    for i in range(LABEL.shape[0]):
        if DR[i] == 0:
            if LABEL[i] == 0:
                y_pred.append(0)
            if LABEL[i] == 1:
                y_pred.append(5)
        if DR[i] == 1:
            if LABEL[i] == 0:
                y_pred.append(1)
            if LABEL[i] == 1:
                y_pred.append(6)
        if DR[i] == 2:
            if LABEL[i] == 0:
                y_pred.append(2)
            if LABEL[i] == 1:
                y_pred.append(7)
        if DR[i] == 3:
            if LABEL[i] == 0:
                y_pred.append(3)
            if LABEL[i] == 1:
                y_pred.append(8)
        if DR[i] == 4:
            if LABEL[i] == 0:
                y_pred.append(4)
            if LABEL[i] == 1:
                y_pred.append(9)
    labels_kemeans = k_means()
    y_true = []
    for i in range(labels_kemeans.shape[0]):
        if DR[i] == 0:
            if labels_kemeans[i] == 0:
                y_true.append(0)
            if labels_kemeans[i] == 1:
                y_true.append(5)
        if DR[i] == 1:
            if labels_kemeans[i] == 0:
                y_true.append(1)
            if labels_kemeans[i] == 1:
                y_true.append(6)
        if DR[i] == 2:
            if labels_kemeans[i] == 0:
                y_true.append(2)
            if labels_kemeans[i] == 1:
                y_true.append(7)
        if DR[i] == 3:
            if labels_kemeans[i] == 0:
                y_true.append(3)
            if labels_kemeans[i] == 1:
                y_true.append(8)
        if DR[i] == 4:
            if labels_kemeans[i] == 0:
                y_true.append(4)
            if labels_kemeans[i] == 1:
                y_true.append(9)
    y_true = y_pred
    cfm = confusion_matrix(y_pred, y_true)

    ## Get Class Labels
    labels = ['DR0_0', 'DR1_0', 'DR2_0', 'DR3_0', 'DR4_0', 'DR0_1', 'DR1_1', 'DR2_1', 'DR3_1', 'DR4_1']
    class_names = labels

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(12, 8))
    ax= plt.subplot()
    sns.heatmap(cfm, annot=True, ax = ax, fmt = 'g', cmap=cm.binary); #annot=True to annotate cells
    # labels, title and ticks

    ax.xaxis.set_label_position('bottom')

    ax.xaxis.set_ticklabels(class_names, fontsize = 14)
    ax.xaxis.tick_bottom()
    ax.yaxis.set_ticklabels(class_names, fontsize = 14)
    ax.set_xlabel('CNN label', fontsize=20, fontweight="bold")
    ax.set_ylabel('CNN label', fontsize=20, fontweight="bold")
    plt.savefig('test_cfmatrix.png')

def plot_hist3d():
    path = 'csv/kaggel_total_reduced.csv'
    csv1 = pd.read_csv(path)
    csv1 = csv1.sample(frac=1).reset_index(drop=True)
    SNR = csv1['SNR'].to_numpy()
    BLUR = csv1['Blur'].to_numpy()
    CONTRAST = csv1['Contrast'].to_numpy()
    BRISQUE = csv1['Brisque'].to_numpy()
    BRIGHT = csv1['Brightness'].to_numpy()
    LABEL = csv1['Qlabel'].to_numpy()
    DR = csv1['DR'].to_numpy()

    fig = plt.figure()
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.set_dpi(96)
    ax = fig.add_subplot(projection='3d')
    # hist, xedges, yedges = np.histogram2d(BRISQUE, LABEL, bins=(20, 20), range=[[50, 100], [0, 1]])
    # hist, xedges, yedges = np.histogram2d(BRIGHT, LABEL, bins=(20, 20), range=[[50, 100], [0, 1]])
    # hist, xedges, yedges = np.histogram2d(BLUR, LABEL, bins=(20, 20), range=[[0, 30], [0, 1]])
    # hist, xedges, yedges = np.histogram2d(SNR, LABEL, bins=(20, 20), range=[[0.7, 2], [0, 1]])
    hist, xedges, yedges = np.histogram2d(DR, LABEL, bins=(20, 20), range=[[0, 4], [0, 1]])
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.1 * np.ones_like(zpos)
    dy = 0.02 * np.ones_like(zpos)
    dz = hist.ravel()
    norm = Normalize()
    colors = cm.jet(norm(dz * 0.1))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
    ax.set_xlabel('SNR', fontweight="bold", fontsize=14)
    ax.set_ylabel('CNN LABEL', fontweight="bold", fontsize=14)
    ax.set_zlabel('No. Images', fontweight="bold", fontsize=14)
    ax.set_yticks([0, 1])
    # ax.set_box_aspect((np.ptp(xpos), 10, 10))
    ax.set_box_aspect((10, 10, 20))
    plt.savefig('images/hist_label_DR.png')

def main():
    path = 'csv/public_datasets.csv'
    csv1 = pd.read_csv(path)
    csv1 = csv1.sample(frac=1).reset_index(drop=True)
    SNR = csv1['SNR'].to_numpy()
    BLUR = csv1['Blur'].to_numpy()
    CONTRAST = csv1['Contrast'].to_numpy()
    BRISQUE = csv1['Brisque'].to_numpy()
    BRIGHT = csv1['Brightness'].to_numpy()
    # LABEL = csv1['Qlabel'].to_numpy()
    # DR = csv1['DR'].to_numpy()
    n = 4000
    snr = []
    blur = []
    contrast = []
    brig = []
    bris = []
    label = []
    matplotlib.rcParams.update({'font.size': 16})
    np.random.seed(10)
    for i in range(n):
        pick = np.random.randint(0, SNR.shape[0])
        snr.append(SNR[pick])  # C
        blur.append(BLUR[pick])  # B
        contrast.append(CONTRAST[pick] ) # SNR
        brig.append(BRIGHT[pick] ) # BRISQUE
        bris.append(BRISQUE[pick] ) # BRISQUE
        # label.append(LABEL[pick] ) # QUALITY LABEL
    # k = 0
    # for i in range(DR.shape[0]):
    #     if DR[i] == 0:
    #         pick = i
    #         snr.append(SNR[pick])  # C
    #         blur.append(BLUR[pick])  # B
    #         contrast.append(CONTRAST[pick] ) # SNR
    #         brig.append(BRIGHT[pick] ) # BRISQUE
    #         bris.append(BRISQUE[pick] ) # BRISQUE
    #         label.append(LABEL[pick] ) # QUALITY LABEL
    #         k += 1
    #     if k == 2000:
    #         break
    # print(len(np.array(blur)))
    # ndim = np.array(snr).shape[0]
    
    # plot_3D(np.array(snr), np.array(blur), np.array(contrast), np.array(bris), 'SNR', '  BLUR', 'CONTRAST', 'BRISQUE', cm.jet_r, 'images/BRISQUE4D')
    # plot_3D(snr, blur, contrast, brig, 'SNR', 'BLUR', 'CONTRAST', 'BRIGHTNESS', cm.YlOrBr_r, 'images/BRIGHTNESS4D')
    # plot_3D(snr, blur, contrast, label, 'SNR', 'BLUR', 'CONTRAST', 'CNN LABEL', cm.Blues, 'images/CNNLABEL4D')

    # plot_2D(snr, blur, bris, 'SNR', 'BLUR', 'BRISQUE', cm.jet_r, 'images/SNR_BLUR_BRISQUE')
    # plot_2D(snr, contrast, bris, 'SNR', 'CONTRAST', 'BRISQUE', cm.jet_r, 'images/SNR_CONTRAST_BRISQUE')
    # plot_2D(blur, contrast, bris, 'BLUR', 'CONTRAST', 'BRISQUE', cm.jet_r, 'images/BLUR_CONTRAST_BRISQUE')
    # plot_2D(brig, contrast, bris, 'BRIGHTNESS', 'CONTRAST', 'BRISQUE', cm.jet_r, 'images/BRIGHTNESS_CONTRAST_BRISQUE')

    # plot_2D(snr, blur, brig, 'SNR', 'BLUR', 'BRIGHTNESS', cm.YlOrBr_r, 'images/SNR_BLUR_BRIGHTNESS')
    # plot_2D(snr, contrast, brig, 'SNR', 'CONTRAST', 'BRIGHTNESS', cm.YlOrBr_r, 'images/SNR_CONTRAST_BRIGHTNESS')
    # plot_2D(blur, contrast, brig, 'BLUR', 'CONTRAST', 'BRIGHTNESS', cm.YlOrBr_r, 'images/BLUR_CONTRAST_BRIGHTNESS')
    # plot_2D(bris, brig, brig, 'BRISQUE', 'BRIGHTNESS', 'BRIGHTNESS', cm.YlOrBr_r, 'images/BRISQUE_BRIGHTNESS')

    # plot_2D(snr, blur, label, 'SNR', 'BLUR', 'CNNLABEL', cm.Blues, 'images/SNR_BLUR_CNNLABEL', tickss=range(2), shrink=0.2, aspect=2)
    # plot_2D(snr, contrast, label, 'SNR', 'CONTRAST', 'CNNLABEL', cm.Blues, 'images/SNR_CONTRAST_CNNLABEL', tickss=range(2), shrink=0.2, aspect=2)
    # plot_2D(blur, contrast, label, 'BLUR', 'CONTRAST', 'CNNLABEL', cm.Blues, 'images/BLUR_CONTRAST_CNNLABEL', tickss=range(2), shrink=0.2, aspect=2)   

    plot_norm_data(BRISQUE, 'BRSIQUE', 75000, [0, 100], np.max(BRISQUE))
    plot_norm_data(SNR, 'SNR', 1200, [0, np.max(SNR)], np.max(SNR))
    plot_norm_data(BRIGHT, 'BRIGHTNESS', 100000, [0, 150], np.max(BRIGHT))
    plot_norm_data(CONTRAST, 'CONTRAST', 500, [0, 1], np.max(CONTRAST))
    BLUR.sort()
    print(np.max(BLUR[:38000]))
    plot_norm_data(BLUR[:38000], 'BLUR', 45000, [0, 55], 55)

    # k_means()

    # plot_hist3d()

if __name__ == '__main__':
    main()
