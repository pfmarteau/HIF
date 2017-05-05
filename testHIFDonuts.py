__author__ = 'P-F.M., February 2017'

import sys
import igraph as ig
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import random as rn
import hif
from sklearn import svm
import copy
import seaborn as sb
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

plt.gcf().subplots_adjust(bottom=0.15)

sb.set_style(style="whitegrid")
sb.set_color_codes()
ig.__version__


matplotlib.rcParams.update({'font.size': 22})

def gen_tore_vecs(dims, number, rmin, rmax):
    vecs = np.random.uniform(low=-1, size=(number,dims))
    radius = rmin + np.random.sample(number) * (rmax-rmin)
    mags = np.sqrt((vecs*vecs).sum(axis=-1))
    # How to distribute the magnitude to the vectors
    for i in range(number):
        vecs[i,:] = vecs[i, :] / mags[i] *radius[i]
    return vecs[:,0], vecs[:,1]



def testOneClassSVM(NU=.1, GAMMA=.1):
    # load data
    f = open('donnutsData.pkl', 'rb')
    [Xn, Xnl, Xnt, Xntl, Xa, Xal, XaLearn, XalLearn, Xb, Xbl, Xc, Xcl] = pickle.load(f)
    f.close()

    N = np.shape(Xn)[0]
    Nt=np.shape(Xnt)[0]
    print("# train normal = ", N)
    print("# test normal = ", Nt)
    Na=np.shape(Xa)[0]
    print("# outliersA = ", Na)


    Nb=np.shape(Xa)[0]
    print("# outliersB = ", Nb)
    Nc=np.shape(Xa)[0]
    print("# outliersC = ", Nc)
    X_train = Xn
    X_test = Xnt
    rng = np.random.RandomState(42)

    # fit the model
    clf = svm.OneClassSVM(nu=NU, kernel="rbf", gamma=GAMMA)

    clf.fit(X_train)

    X_T = np.r_[X_test, Xa, Xb, Xc]
    X_Ta = np.r_[X_test, Xa]
    X_Tb = np.r_[X_test, Xb]
    X_Tc = np.r_[X_test, Xc]
    scoring = clf.fit(X_train).decision_function(X_T)
    scoringa = clf.fit(X_train).decision_function(X_Ta)
    scoringb = clf.fit(X_train).decision_function(X_Tb)
    scoringc = clf.fit(X_train).decision_function(X_Tc)
    y_true = np.array([1] * (N) + [-1] * (Na+Nb+Nc))
    y_truea = np.array([1] * (N) + [-1] * (Na))
    y_trueb = np.array([1] * (N) + [-1] * (Nb))
    y_truec = np.array([1] * (N) + [-1] * (Nc))
    fpr, tpr, thresholds = roc_curve(y_true, scoring)
    roc_auc = auc(fpr, tpr)
    fpra, tpra, thresholds = roc_curve(y_truea, scoringa)
    roc_auca = auc(fpr, tpr)
    fprb, tprb, thresholds = roc_curve(y_trueb, scoringb)
    roc_aucb = auc(fpr, tpr)
    fprc, tprc, thresholds = roc_curve(y_truec, scoringc)
    roc_aucc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    f = open('SYN-1C-SVM-ROC.pkl', 'wb')
    pickle.dump([fpr,tpr,fpra,tpra,fprb,tprb,fprc,tprc], f)
    f.close()

    return roc_auc, roc_auca, roc_aucb, roc_aucc

def testTwoClassSVM(C=.1, GAMMA=.1):
    # load data
    f = open('donnutsData.pkl', 'rb')
    [Xn, Xnl, Xnt, Xntl, Xa, Xal, XaLearn, XalLearn, Xb, Xbl, Xc, Xcl] = pickle.load(f)
    f.close()

    N = np.shape(Xn)[0]
    Nt=np.shape(Xnt)[0]
    print("# train normal = ", N)
    print("# test normal = ", Nt)
    Na=np.shape(Xa)[0]
    print("# outliersA = ", Na)
    NaL=np.shape(XaLearn)[0]
    print("# outliersA for learning = ", NaL)
    Nb=np.shape(Xa)[0]
    print("# outliersB = ", Nb)
    Nc=np.shape(Xa)[0]
    print("# outliersC = ", Nc)
    X_train = np.concatenate((Xn, XaLearn), axis=0)
    X_test = Xnt

    # fit the model
    clf = svm.SVC(C=C, kernel="rbf", gamma=GAMMA, probability=True)
    y_train = np.array([1] * (N) + [-1] * (NaL))
    '''clf.fit(X_train, y_train)'''

    X_T = np.r_[X_test, Xa, Xb, Xc]
    X_Ta = np.r_[X_test, Xa]
    X_Tb = np.r_[X_test, Xb]
    X_Tc = np.r_[X_test, Xc]
    scoring = clf.fit(X_train,y_train).decision_function(X_T)
    scoringa = clf.fit(X_train,y_train).decision_function(X_Ta)
    scoringb = clf.fit(X_train,y_train).decision_function(X_Tb)
    scoringc = clf.fit(X_train,y_train).decision_function(X_Tc)
    y_true = np.array([1] * (N) + [-1] * (Na+Nb+Nc))
    y_truea = np.array([1] * (N) + [-1] * (Na))
    y_trueb = np.array([1] * (N) + [-1] * (Nb))
    y_truec = np.array([1] * (N) + [-1] * (Nc))
    fpr, tpr, thresholds = roc_curve(y_true, scoring)
    roc_auc = auc(fpr, tpr)
    fpra, tpra, thresholds = roc_curve(y_truea, scoringa)
    roc_auca = auc(fpr, tpr)
    fprb, tprb, thresholds = roc_curve(y_trueb, scoringb)
    roc_aucb = auc(fpr, tpr)
    fprc, tprc, thresholds = roc_curve(y_truec, scoringc)
    roc_aucc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    f = open('SYN-2C-SVM-ROC.pkl', 'wb')
    pickle.dump([fpr,tpr,fpra,tpra,fprb,tprb,fprc,tprc], f)
    f.close()

    return roc_auc, roc_auca, roc_aucb, roc_aucc



def normalize(Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2):
    mx0=max(max(Snt0), max(Sa0), max(Sb0), max(Sc0))
    mx1 = max(max(Snt1), max(Sa1), max(Sb1), max(Sc1))
    mx2 = max(max(Snt2), max(Sa2), max(Sb2), max(Sc2))
    mn0=min(min(Snt0), min(Sa0), min(Sb0), min(Sc0))
    mn1 = min(min(Snt1), min(Sa1), min(Sb1), min(Sc1))
    mn2 = min(min(Snt2), min(Sa2), min(Sb2), min(Sc2))
    print('mx0='+str(mx0)+' mx1='+str(mx1)+' mx2='+str(mx2))
    print('mn0=' + str(mn0) + ' mn1=' + str(mn1)+' mx2='+str(mx2))

    if (mx0 != mn0):
        Snt0= (Snt0-mn0) / (mx0-mn0)
        Sa0 = (Sa0 - mn0) /(mx0-mn0)
        Sb0 = (Sb0 - mn0) /(mx0-mn0)
        Sc0 = (Sc0 - mn0) /(mx0-mn0)


    if (mx1 != mn1):
        Snt1 = (Snt1 - mn1) / (mx1 -mn1)
        Sa1 = (Sa1 - mn1) / (mx1 - mn1)
        Sb1 = (Sb1 - mn1) / (mx1 - mn1)
        Sc1 = (Sc1 - mn1) / (mx1 - mn1)

    if(mx2 != mn2):
        Snt2 = (Snt2 - mn2) / (mx2 - mn2)
        Sa2 = (Sa2 - mn2) / (mx2 - mn2)
        Sb2 = (Sb2 - mn2) / (mx2 - mn2)
        Sc2 = (Sc2 - mn2) / (mx2 - mn2)

    return Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2





def plotGlobalAucBis(contamin=True):
    plt.close('all')

    [Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2] = np.load('dataHIF.npy')
    [Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2] = normalize(Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2)

    Nt = len(Snt0)
    Na = len(Sa0)
    Nb = len(Sb0)
    Nc = len(Sc0)
    y_true = np.array([-1] * Nt + [1] * (Na+Nb+Nc))
    y_truea = np.array([-1] * Nt + [1] * (Na))
    y_trueb = np.array([-1] * Nt + [1] * (Nb))
    y_truec = np.array([-1] * Nt + [1] * (Nc))

    nb=21
    x=np.linspace(0,1,nb)
    if(contamin):
        x1=np.linspace(0,1,nb)
    else:
        x1=[0]

    print(x1)
    Tauc=np.zeros((len(x), len(x1)))


    Alpha1_auca = np.zeros(len(x))
    Alpha1_aucb = np.zeros(len(x))
    Alpha1_aucc = np.zeros(len(x))

    i=0
    for alpha1 in x:
        S_n1 = alpha1 * Snt0 + (1 - alpha1) * Snt1
        S_a1 = alpha1 * Sa0 + (1 - alpha1) * Sa1
        S_b1 = alpha1 * Sb0 + (1 - alpha1) * Sb1
        S_c1 = alpha1 * Sc0 + (1 - alpha1) * Sc1
        fpr, tpr, thresholds = roc_curve(y_truea, np.r_[S_n1, S_a1])
        roc_aucc = auc(fpr, tpr)
        Alpha1_auca[i] = roc_aucc
        fpr, tpr, thresholds = roc_curve(y_truea, np.r_[S_n1, S_b1])
        roc_aucc = auc(fpr, tpr)
        Alpha1_aucb[i] = roc_aucc
        fpr, tpr, thresholds = roc_curve(y_truea, np.r_[S_n1, S_c1])
        roc_aucc = auc(fpr, tpr)
        Alpha1_aucc[i] = roc_aucc
        i+=1

    plt.figure(1)
    plt.plot(x, Alpha1_auca, color='r', marker='s')
    plt.plot(x, Alpha1_aucb, color='g', marker='d')
    plt.plot(x, Alpha1_aucc, color='c', marker='*')
    plt.ylabel('AUC', fontsize=20)
    plt.xlabel(r'$\alpha_1$', fontsize=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_AUC_alpha1.eps')

    Alpha2_auca = np.zeros(len(x))
    Alpha2_aucb = np.zeros(len(x))
    Alpha2_aucc = np.zeros(len(x))
    i=0
    for alpha2 in x:
        S_n2 = alpha2 * Snt0 + (1 - alpha2) * Snt2
        S_a2 = alpha2 * Sa0 + (1 - alpha2) * Sa2
        S_b2 = alpha2 * Sb0 + (1 - alpha2) * Sb2
        S_c2 = alpha2 * Sc0 + (1 - alpha2) * Sc2

        fpr, tpr, thresholds = roc_curve(y_truea, np.r_[S_n2, S_a2])
        roc_aucc = auc(fpr, tpr)
        Alpha2_auca[i] = roc_aucc
        fpr, tpr, thresholds = roc_curve(y_trueb, np.r_[S_n2, S_b2])
        roc_aucc = auc(fpr, tpr)
        Alpha2_aucb[i] = roc_aucc
        fpr, tpr, thresholds = roc_curve(y_truec, np.r_[S_n2, S_c2])
        roc_aucc = auc(fpr, tpr)
        Alpha2_aucc[i] = roc_aucc

        i+=1

    plt.figure(2)
    plt.plot(x, Alpha2_auca, color='r', marker='s')
    plt.plot(x, Alpha2_aucb, color='g', marker='d')
    plt.plot(x, Alpha2_aucc, color='c', marker='*')
    plt.ylabel('AUC', fontsize=20)
    plt.xlabel(r'$\alpha_2$', fontsize=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_AUC_alpha2.eps')


    Sa0 = np.concatenate((Sa0,Sb0), axis=0)
    Sa0 = np.concatenate((Sa0, Sc0), axis=0)

    Sa1 = np.concatenate((Sa1,Sb1), axis=0)
    Sa1 = np.concatenate((Sa1, Sc1), axis=0)

    Sa2 = np.concatenate((Sa2,Sb2), axis=0)
    Sa2 = np.concatenate((Sa2, Sc2), axis=0)

    Alpha1_auc = np.zeros(len(x))
    Zauc=[]
    i=0
    for alpha1 in x:
        j=0
        S_n1 = alpha1 * Snt0 + (1 - alpha1) * Snt1
        S_a1 = alpha1 * Sa0 + (1 - alpha1) * Sa1
        fpr, tpr, thresholds = roc_curve(y_true, np.r_[S_n1, S_a1])
        roc_aucc = auc(fpr, tpr)
        Alpha1_auc[i] = roc_aucc


        for alpha2 in x1:
            S_n2 = alpha2 * S_n1 + (1-alpha2) * Snt2
            S_a2 = alpha2 * S_a1 + (1-alpha2) * Sa2
            fpr, tpr, thresholds = roc_curve(y_true, np.r_[S_n2, S_a2])
            Tauc[i][j] = auc(fpr, tpr)
            Zauc.append([alpha1,alpha2,Tauc[i][j]])
            j = j+1
        i=i+1

    fig=plt.figure(10)
    ax = fig.gca(projection='3d')
    if(contamin):
        i, j = np.unravel_index(Tauc.argmax(), Tauc.shape)
        print('(alpha1*, alpha2*)', x[i], x1[j], 'AUC*', Tauc[i][j])
        Zauc = np.array(Zauc)
        X, Y = np.meshgrid(Zauc[:,0], Zauc[:,1])
        ax.plot_trisurf(Zauc[:, 0], Zauc[:, 1], Zauc[:, 2], cmap=cm.jet, linewidth=0)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax.set_xlabel(r"$\alpha_1$", fontsize=20, labelpad=20)
        ax.set_ylabel(r'$\alpha_2$', fontsize=20, labelpad=20)
        ax.set_zlabel('AUC', fontsize=20, labelpad=20)
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.savefig('syn_HIF1_AUC_alpha1_Alpha2.eps')
        plt.show()

    plt.figure(11)
    plt.plot(x, Alpha1_auc, color='b', marker='o')
    plt.ylabel('AUC', fontsize=20)
    plt.xlabel(r'$\alpha_1$', fontsize=20)

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_AUC_all_alpha1.eps')

    i=np.argmax(Alpha1_auc)
    print('(alpha0*)', x[i], 'AUC*', Alpha1_auc[i])
    fpr, tpr, thresholds = roc_curve(y_true, np.r_[Snt0, Sa0])
    roc_aucc = auc(fpr, tpr)
    print("AUC IF", roc_aucc)
    return Alpha1_auc




def plotDetailedResults(alpha0=.5, alpha1=.5, alpha2=.5):

    plt.close('all')

    [Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2] = np.load('dataHIF.npy')
    [Snt0, Snt1, Snt1, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2] = normalize(Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2)

    St_0 = alpha0 * Snt0 + (1 - alpha0) * Snt1
    Sa_0 = alpha0 * Sa0 + (1 - alpha0) * Sa1
    Sb_0 = alpha0 * Sb0 + (1 - alpha0) * Sb1
    Sc_0 = alpha0 * Sc0 + (1 - alpha0) * Sc1


    St1 = alpha1 * Snt0 + (1 - alpha1) * Snt1
    Sa1 = alpha1 * Sa0 + (1 - alpha1) * Sa1
    Sb1 = alpha1 * Sb0 + (1 - alpha1) * Sb1
    Sc1 = alpha1 * Sc0 + (1 - alpha1) * Sc1

    St2 = alpha2 * St1 + (1 - alpha2) * Snt1
    Sa2 = alpha2 * Sa1 + (1 - alpha2) * Sa1
    Sb2 = alpha2 * Sb1 + (1 - alpha2) * Sb1
    Sc2 = alpha2 * Sc1 + (1 - alpha2) * Sc1

    sb.set(font_scale=1.5)

    Nt = len(Snt0)
    Na = len(Sa0)
    Nb = len(Sb0)
    Nc = len(Sc0)
    y_true = np.array([-1] * Nt + [1] * (Na+Nb+Nc))
    y_truea = np.array([-1] * Nt + [1] * (Na))
    y_trueb = np.array([-1] * Nt + [1] * (Nb))
    y_truec = np.array([-1] * Nt + [1] * (Nc))

    plt.figure(1)

    # Plot path length distribution
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(Snt0, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa0, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.xlabel('anomaly score for normal (blue) and anomaly (red) data', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribIF1.eps')
    plt.show()

    plt.figure(2)
    # Plot path length distribution
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(Snt0, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa0, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb0, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red and green) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribIF11.eps')
    plt.show()

    plt.figure(3)
    # Plot path length distribution
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(Snt0, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa0, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb0, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red and cyan) data')
    sb.distplot(Sc0, kde=True, color="c", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red, green and cyan) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribIF111.eps')

    plt.figure(4)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St1, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa1, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF1.eps')

    plt.figure(5)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St1, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa1, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb1, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red, green and cyan) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF11.eps')

    plt.figure(6)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St1, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa1, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb1, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (green) data')
    sb.distplot(Sc1, kde=True, color="c", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red, green and cyan) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF111.eps')

    plt.figure(7)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St2, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa2, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF2.eps')

    plt.figure(8)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St2, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa2, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb2, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red, green and cyan) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF21.eps')

    plt.figure(9)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(St2, kde=True, color="b", ax=axes, axlabel='anomaly score for normal data')
    sb.distplot(Sa2, kde=True, color="r", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red) data')
    sb.distplot(Sb2, kde=True, color="g", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (green) data')
    sb.distplot(Sc2, kde=True, color="c", ax=axes, axlabel='anomaly score for normal (blue) and anomaly (red, green and cyan) data')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('distribHIF211.eps')

    plt.show()
    plt.close('all')

    '''mx0 = max(1.0, max(Snt0), max(Sa0), max(Sb0), max(Sc0))
    rca0 = evalROCcurve(Snt0, Sa0, [], max=mx0)
    rcb0 = evalROCcurve(Snt0, Sb0, [], max=mx0)
    rcc0 = evalROCcurve(Snt0, Sc0, [], max=mx0)

    mx = max(1.0, max(St1), max(Sa1), max(Sb1), max(Sc1))
    rca1 = evalROCcurve(St1, Sa1, [], max=mx)
    rcb1 = evalROCcurve(St1, Sb1, [], max=mx)
    rcc1 = evalROCcurve(St1, Sc1, [], max=mx)

    mx = max(1.0, max(St2), max(Sa2), max(Sb2), max(Sc2))
    rca2 = evalROCcurve(St2, Sa2, [], max=mx)
    rcb2 = evalROCcurve(St2, Sb2, [], max=mx)
    rcc2 = evalROCcurve(St2, Sc2, [], max=mx)'''

    f = open('SYN-1C-SVM-ROC.pkl', 'rb')
    [fpr1, tpr1, fpra1, tpra1, fprb1, tprb1, fprc1, tprc1]=pickle.load( f)
    f.close()
    f = open('SYN-2C-SVM-ROC.pkl', 'rb')
    [fpr2, tpr2, fpra2, tpra2, fprb2, tprb2, fprc2, tprc2]=pickle.load( f)
    f.close()

    plt.figure(10)
    fpr_a0, tpr_a0, thresholds = roc_curve(y_truea, np.r_[Snt0, Sa0])
    plt.plot(fpr_a0,tpr_a0, 'r', linewidth=2)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_IF_ROC_scores_a.eps')

    plt.figure(11)
    fpr_b0, tpr_b0, thresholds = roc_curve(y_trueb, np.r_[Snt0, Sb0])
    plt.plot(fpr_b0, tpr_b0, 'r', linewidth=2)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_IF_ROC_scores_b.eps')

    plt.figure(20)
    plt.plot(fpr_a0,tpr_a0, ':r', label='IF', linewidth=2)
    fpr_a1, tpr_a1, thresholds = roc_curve(y_truea, np.r_[St1, Sa1])
    plt.plot(fpr_a1, tpr_a1, '--r', label='HIF1', linewidth=2)
    fpr_a2, tpr_a2, thresholds = roc_curve(y_truea, np.r_[St2, Sa2])
    plt.plot(fpr_a2, tpr_a2, '-.r', label='HIF2', linewidth=2)
    plt.plot(fpra1,tpra1, '-r', label='1C-SVM', linewidth=1)
    plt.plot(fpra2,tpra2, '-or', label='2C-SVM', linewidth=1)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.legend()
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_ROC_scores_a.eps')

    plt.figure(21)
    plt.plot(fpr_b0, tpr_b0, ':g', label='IF', linewidth=2)
    fpr_b1, tpr_b1, thresholds = roc_curve(y_trueb, np.r_[St1, Sb1])
    plt.plot(fpr_b1, tpr_b1, '--g', label='HIF1', linewidth=2)
    fpr_b2, tpr_b2, thresholds = roc_curve(y_trueb, np.r_[St2, Sb2])
    plt.plot(fpr_b2, tpr_b2, '-.g', label='HIF2', linewidth=2)
    plt.plot(fprb1,tprb1, '-g', label='1C-SVM', linewidth=1)
    plt.plot(fprb2,tprb2, '-og', label='2C-SVM', linewidth=1)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.legend()
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_ROC_scores_b.eps')

    plt.figure(22)
    fpr_c0, tpr_c0, thresholds = roc_curve(y_truec, np.r_[Snt0, Sc0])
    plt.plot(fpr_c0, tpr_c0, ':c', label='IF', linewidth=2)
    fpr_c1, tpr_c1, thresholds = roc_curve(y_truec, np.r_[St1, Sc1])
    plt.plot(fpr_c1, tpr_c1, '--c', label='HIF1', linewidth=2)
    fpr_c2, tpr_c2, thresholds = roc_curve(y_truec, np.r_[St2, Sc2])
    plt.plot(fpr_c2, tpr_c2, '-.c', label='HIF2', linewidth=2)
    plt.plot(fprc1,tprc1, '-c', label='1C-SVM', linewidth=1)
    plt.plot(fprc2,tprc2, '-oc', label='2C-SVM', linewidth=1)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.legend()
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('syn_HIF_ROC_scores_c.eps')
    plt.show()

    print('AUC IF (a) = ', auc(fpr_a0, tpr_a0))
    print('AUC IF (b) = ', auc(fpr_b0, tpr_b0))
    print('AUC IF (c) = ', auc(fpr_c0, tpr_c0))
    print('AUC HIF1 (a) = ', auc(fpr_a1, tpr_a1))
    print('AUC HIF1 (b) = ', auc(fpr_b1, tpr_b1))
    print('AUC HIF1 (c) = ', auc(fpr_c1, tpr_c1))
    print('AUC HIF2 (a) = ', auc(fpr_a2, tpr_a2))
    print('AUC HIF2 (b) = ', auc(fpr_b2, tpr_b2))
    print('AUC HIF2 (c) = ', auc(fpr_c2, tpr_c2))



def createDonutData(contamin=.005):
    Nobjs = 1000
    xn, yn = gen_tore_vecs(2, Nobjs, 1.5, 4)
    #xn, yn = gen_tore_vecs(2, Nobjs, 0, 4)
    Xn = np.array([xn, yn]).T
    Xnl = np.zeros(Nobjs)
    '''pca = PCA(n_components=2, whiten=True)
    pca.fit(Xno)
    Xn=pca.transform(Xno)'''

    Nobjsb = 1000
    mean = [0, 0]
    cov = [[.5, 0], [0, .5]]  # diagonal covariance
    xb, yb = np.random.multivariate_normal(mean, cov, Nobjsb).T
    Xb = np.array([xb, yb]).T
    Xbl = np.ones(Nobjsb)*3

    Nobjst = 1000
    xnt, ynt = gen_tore_vecs(2, Nobjst, 1.5, 4)
    Xnt = np.array([xnt, ynt]).T
    Xntl = np.zeros(Nobjst)

    # create cluster of anomalies
    mean = [3., 3.]
    cov = [[.25, 0], [0, .25]]  # diagonal covariance
    Nobjsa = 1000
    xa, ya = np.random.multivariate_normal(mean, cov, Nobjsa).T
    Xa = np.array([xa, ya]).T
    Xal = np.ones(Nobjsa)

    NobjsaL = int(1000*contamin)
    xaLearn, yaLearn = np.random.multivariate_normal(mean, cov, NobjsaL).T
    XaLearn = np.array([xaLearn, yaLearn]).T
    XalLearn = np.ones(Nobjsa)

    mean = [-3., -3.]
    cov = [[.25, 0], [0, .25]]  # diagonal covariance
    Nobjsc = 1000
    xc, yc = np.random.multivariate_normal(mean, cov, Nobjsc).T
    Xc = np.array([xc, yc]).T
    Xcl = 2*np.ones(Nobjsc)

    f = open('donnutsData.pkl', 'wb')
    pickle.dump([Xn, Xnl, Xnt, Xntl,Xa, Xal, XaLearn, XalLearn, Xb, Xbl, Xc, Xcl], f)
    f.close()

def computeHIF(ntrees=1024, sample_size=32):

    # load data
    f = open('donnutsData.pkl', 'rb')
    [Xn, Xnl, Xnt, Xntl, Xa, Xal, XaLearn, XalLearn, Xb, Xbl, Xc, Xcl] = pickle.load(f)
    f.close()

    Nobjs = len(Xn)
    Nobjst = len(Xnt)
    Nobjsa = len(Xa)
    NobjsaL = len(XaLearn)
    print('N anomaly added:', NobjsaL)
    Nobjsb = len(Xb)
    Nobjsc = len(Xc)
    xn=Xn[:,0]
    yn=Xn[:,1]
    xa=Xa[:,0]
    ya=Xa[:,1]
    xaLearn=XaLearn[:,0]
    yaLearn=XaLearn[:,1]
    xb=Xb[:,0]
    yb=Xb[:,1]
    xc=Xc[:,0]
    yc=Xc[:,1]

    X = Xn

    plt.figure(1)
    plt.plot(xn, yn, 'bo', markersize=10)
    plt.savefig('clustersHIF0.eps')

    nn=100
    plt.figure(2)
    plt.plot(xn, yn, 'bo', xa[0:nn], ya[0:nn], 'rs', markersize=10)
    plt.savefig('clustersHIF1.eps')


    plt.figure(3)
    plt.plot(xn, yn, 'bo', xa[0:nn], ya[0:nn], 'rs', xb[0:nn], yb[0:nn], 'gd', markersize=12)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.savefig('clustersHIF2.eps')

    plt.figure(4)
    plt.plot(xn, yn, 'bo', xa[0:nn], ya[0:nn], 'rs', xb[0:nn], yb[0:nn], 'gd', xc[0:nn], yc[0:nn], 'c*', markersize=12)
    plt.plot(xaLearn, yaLearn, 'ks', markersize=10)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.savefig('clustersHIF3.eps')

    # Creating Forest on normal data + anomalies labels
    print('building the HIF ...')

    F = hif.hiForest(X, ntrees=ntrees, sample_size=sample_size)

    # add anomaly labels in isolation buckets
    for i in range(NobjsaL):
            F.addAnomaly(XaLearn[i], XalLearn[i])
    F.computeAnomalyCentroid()
    f = open('DonnutsHIF.pkl', 'wb')
    pickle.dump(F, f)
    f.close()

    # Computing path for all normal test data
    print('evaluating normal test instances ...')
    Snt0 = np.zeros(Nobjst)
    Snt1 = np.zeros(Nobjst)
    Snt2 = np.zeros(Nobjst)
    for i in range(Nobjst):
        s, l, d, dc = F.computeAggScore(Xnt[i])
        Snt0[i] = s
        Snt1[i] = d
        Snt2[i] = dc


    # Computing path for anomaly data points
    print('evaluating anomaly data (red) ...')
    Sa0 = np.zeros(Nobjsa)
    Sa1 = np.zeros(Nobjsa)
    Sa2 = np.zeros(Nobjsa)
    for i in range(Nobjsa):
        s, l, d, dc = F.computeAggScore(Xa[i])
        Sa0[i] = s
        Sa1[i] = d
        Sa2[i] = dc

    # Computing path for new anomaly data points
    print('evaluating anomaly data (green) ...')
    Sb0 = np.zeros(Nobjsb)
    Sb1 = np.zeros(Nobjsb)
    Sb2 = np.zeros(Nobjsb)
    for i in range(Nobjsb):
        s, l, d, dc = F.computeAggScore(Xb[i])
        Sb0[i] = s
        Sb1[i] = d
        Sb2[i] = dc


    # Computing path for new anomaly data points
    print('evaluating anomaly data (cyan) ...')
    Sc0 = np.zeros(Nobjsc)
    Sc1 = np.zeros(Nobjsc)
    Sc2 = np.zeros(Nobjsc)
    for i in range(Nobjsc):
        s, l, d, dc = F.computeAggScore(Xc[i])
        Sc0[i] = s
        Sc1[i] = d
        Sc2[i] = dc


    np.save('dataHIF.npy', [Snt0, Snt1, Snt2, Sa0, Sa1, Sa2, Sb0, Sb1, Sb2, Sc0, Sc1, Sc2])
    return F


