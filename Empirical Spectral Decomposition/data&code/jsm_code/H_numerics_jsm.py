import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pylab import setp
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import eigsh
import numpy as np
import pickle
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(
            n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

norm = np.linalg.norm
sqrt = np.sqrt

rf_rate = 0.0

plt.rc('ps',usedistiller='xpdf')
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Times']})
params= {'text.latex.preamble' : 
    r'\usepackage[scr=boondox]{mathalfa}'}
plt.rcParams.update(params)

matplotlib.rcParams.update({'font.size': 14})

data = pickle.load(open("_data_jsm.p", "rb"))
Z = data['Z']
P = data['P']

num_sims = data['num_sims']
num_block = data['num_block']
q = data['q']
n = data['n']
paths = np.array([5]) #(data['paths']
pidx = data['pidx']
mus = data['mus']

plim = data['pmax']
if ( plim > 3000):
    plim = 3000

fret = data['fret']
sret = data['sret'][0:plim]
Bret = data['Bret'][0:plim]
alph = data['alpha'][0:plim]
svol = data['svol'][0:plim]
Bvol = data['Bvol'][0:plim]
vols = sqrt(Bvol**2 + svol**2)

F = data['F']
fvol = sqrt(np.diag(F))
C = F * np.outer(1/fvol, 1/fvol)

# print factor variance matrix
file = open("params.tex", "w")
F_tex = "\\begin{align} \n \\var(f) = \\left( \\begin{array}"
F_tex = F_tex + "{" + ''.join(np.repeat('c',q)) + "} \n"
file.writelines(F_tex)
for i in range(np.shape(F)[0]):
    ary = list(map(lambda x : f"${int(x)}$", F[i,:]))
    row = ' & '.join(ary) + ' \\\\ \n'
    file.writelines(row)

F_tex = "\\end{array} \\right) \n \\end{align} \n"
file.writelines(F_tex)

F_tex = "\\begin{align} \n \\text{corr}(f) = \\left( \\begin{array}"
F_tex = F_tex + "{" + ''.join(np.repeat('c',q)) + "} \n"
file.writelines(F_tex)

for i in range(np.shape(C)[0]):
    ary = list(map(lambda x : f"${format(round(x,2)+0,'0.2f')}$", C[i,:]))
    row = ' & '.join(ary) + ' \\\\ \n'
    file.writelines(row)

F_tex = "\\end{array} \\right) \n \\end{align}"
file.writelines(F_tex)

F_tex = "Factor volatilities $\\sigma_f = \\sqrt{\\diag(\\var(f))}$"
F_tex = F_tex + " are given by, \n \\begin{align} \n \\sigma_f"
F_tex = F_tex + " = \\left( \\begin{array}"
F_tex = F_tex + "{" + ''.join(np.repeat('c',q)) + "} \n"
file.writelines(F_tex)
ary = list(map(lambda x : f"${format(round(x,1)+0,'0.1f')}$",fvol))
row = ' & '.join(ary) + ' \\\\ \n'
file.writelines(row)

F_tex = "\\end{array} \\right) \n \\end{align}"
file.writelines(F_tex)

F_tex = "with expectations, "
F_tex = F_tex + "\\begin{align} \n \\Exp(f) = \\left( \\begin{array}"
F_tex = F_tex + "{" + ''.join(np.repeat('c',q)) + "} \n"
file.writelines(F_tex)
ary = list(map(lambda x : f"${format(round(x,2)+0,'0.2f')}$",fret))
row = ' & '.join(ary) + ' \\\\ \n'
file.writelines(row)

F_tex = "\\end{array} \\right) \n \\end{align}"
file.writelines(F_tex)

F_tex = "Population means $\\alpha = \\Exp(y) = B \\Exp(f) + \\Exp(\\ep)$"
F_tex = F_tex + " have the averages,"
F_tex = F_tex + "\\begin{align*} \n"
F_tex = F_tex + "p^{-1} \\tsum_{i=1}^p \\alpha_i = "
F_tex = F_tex + f"{round(np.mean(alph),2)}"
F_tex = F_tex + "\\quad  p^{-1} \\tsum_{i=1}^p (B\\Exp(f))_i = "
F_tex = F_tex + f"{round(np.mean(Bret),2)}"
F_tex = F_tex + "\quad p^{-1} \\tsum_{i=1}^p \\Exp(\ep_i) = "
F_tex = F_tex + f"{round(np.mean(sret),2)}"
F_tex = F_tex + "\n \\end{align*}"

file.writelines(F_tex)
F_tex = "Population volatilities $\\sigma_X =\\sqrt{\\diag(\\var(X))}$"
F_tex = F_tex + " are given by, \n \\begin{align*} \n"
F_tex = F_tex + "p^{-1} \\tsum_{i=1}^p (\\sigma_y)_i = "
F_tex = F_tex + f"{round(np.mean(vols),2)}"
F_tex = F_tex + "\\quad  p^{-1} \\tsum_{i=1}^p (\\sigma_{Bf})_i = "
F_tex = F_tex + f"{round(np.mean(Bvol),2)}"
F_tex = F_tex + "\quad p^{-1} \\tsum_{i=1}^p (\\sigma_{\ep})_i = "
F_tex = F_tex + f"{round(np.mean(svol),2)} \, ."
F_tex = F_tex + "\n \\end{align*}"


file.writelines(F_tex)

file.close()




### Population model images ###
for p_info in P:
    p = p_info['p']

    pr = p_info['pr']
    pv = p_info['pv']
    
    plt.plot(pv, pr)
    plt.title(f"Population Frontier (p = {p})")
    #plt.ylim(0, 0.06)
    #plt.xlim(20, 75)
    plt.tight_layout()
    plt.savefig(f"img/frontier{p}.pdf", 
        transparent=True, format="pdf")
    plt.close()

    if (p == 3000):
        BBp = p_info['XI_block']
        gcm = plt.get_cmap('BuPu') 
        gcm = truncate_colormap(gcm, 0, 1)
        plt.matshow(BBp @ BBp.T, cmap=gcm)
        plt.savefig(f"img/block_orig.pdf", 
            transparent=True, format="pdf")
            #    format="svg")
        plt.close()
        g = csr_matrix(BBp @ BBp.T)
        perm = reverse_cuthill_mckee(g)

        for i in range(num_block):
            BBp[:,i] = BBp[perm,i]
        plt.matshow(BBp @ BBp.T, cmap=gcm)
        plt.savefig(f"img/block_post.pdf", 
            transparent=True, format="pdf")
            #    format="svg")
        plt.close()
        
        for k in paths: # is is for 3K -- fixme
            plt.plot(Z[3][k]['mfs_pv_est'], Z[3][k]['mfs_pr_est'], 'o')
            plt.title(f"Estimated corrected frontier (p = {p})")
            plt.savefig(f"img/est_mfs_frontier_p_{p}.pdf", 
                transparent=True, format="pdf")
            plt.close()
            plt.plot(Z[3][k]['mfs_pv_act'], Z[3][k]['mfs_pr_act'], 'o')
            plt.title(f"Actual corrected frontier (p = {p})")
            plt.savefig(f"img/act_mfs_frontier_p_{p}.pdf", 
                transparent=True, format="pdf")
            plt.tight_layout()
            plt.close()

            plt.plot(Z[3][k]['pca_pv_est'], Z[3][k]['pca_pr_est'], 'o')
            plt.title(f"Estimated PCA frontier (p = {p})")
            plt.savefig(f"img/est_pca_frontier_p_{p}.pdf", 
                transparent=True, format="pdf")
            plt.tight_layout()
            plt.close()

            plt.plot(Z[3][k]['pca_pv_act'], Z[3][k]['pca_pr_act'], 'o')
            plt.title(f"Actual PCA frontier (p = {p})")
            plt.savefig(f"img/act_pca_frontier_p_{p}.pdf", 
                transparent=True, format="pdf")
            plt.tight_layout()
            plt.close()



fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(Bret, bins=40, color="gray", 
    alpha = 0.25, density=True)
#plt.ylim(0, 0.06)
#plt.xlim(20, 75)
plt.tight_layout()
fig.savefig(f"img/Bret_hist.pdf", 
    transparent=True, format="pdf")
plt.close()

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(Bret, bins=40, color="gray", 
    alpha = 0.25, density=True)
#plt.ylim(0, 0.06)
#plt.xlim(20, 75)
plt.tight_layout()
fig.savefig(f"img/Bvol_hist.pdf", 
    transparent=True, format="pdf")
plt.close()

color = 'cornflowerblue'
color = 'lightsteelblue'

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.plot(Bvol, Bret, 'o', alpha = 0.25, color=color)
plt.ylim(-15, 32.5)
plt.xlim(0, 80)
plt.xlabel("Systematic volatility")
plt.ylabel("Systematic return")
plt.yticks(np.arange(-10, 30 + 0.1, 5))
plt.xticks(np.arange(5, 80 + 0.1, 10))
plt.tight_layout()
fig.savefig(f"img/Bvol_Bret.pdf", 
    transparent=True, format="pdf")
plt.close()

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.plot(svol, sret, 'o', alpha = 0.25, color=color)
plt.ylim(-15, 32.25)
plt.xlim(0, 80)
plt.xlabel("Specific volatility")
plt.ylabel("Specific return")
plt.yticks(np.arange(-10, 30 + 0.1, 5))
plt.xticks(np.arange(5, 80 + 0.1, 10))
plt.tight_layout()
fig.savefig(f"img/svol_sret.pdf", 
    transparent=True, format="pdf")
plt.close()




fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.plot(vols, alph, 'o', alpha = 0.25, color=color)
plt.ylim(-15, 32.25)
plt.xlim(0, 80)
plt.xlabel("Security volatility")
plt.ylabel("Security return")
plt.yticks(np.arange(-10, 30 + 0.1, 5))
plt.xticks(np.arange(5, 80 + 0.1, 10))
plt.tight_layout()
fig.savefig(f"img/vols_alph.pdf", 
    transparent=True, format="pdf")
plt.close()





fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(sret, bins=40, color="gray", 
    alpha = 0.25, density=True)
#plt.ylim(0, 0.06)
#plt.xlim(20, 75)
plt.tight_layout()
fig.savefig(f"img/sret_hist.pdf", 
    transparent=True, format="pdf")
plt.close()


fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(svol, bins=40, color="gray", 
    alpha = 0.25, density=True)
#plt.ylim(0, 0.06)
#plt.xlim(20, 75)
plt.tight_layout()
fig.savefig(f"img/svol_hist.pdf", 
    transparent=True, format="pdf")
plt.close()

fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(alph, bins=40, color="gray", 
    alpha = 0.5, density=True)
#plt.ylim(0, 0.06)
#plt.xlim(-5, 25)
plt.tight_layout()
fig.savefig(f"img/alph_hist.pdf", 
    transparent=True, format="pdf")
plt.close()


XI = data['XI']
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.hist(XI[:,0], bins=40, color=color, 
    density=True, label="1", alpha = 0.75)
ax.hist(XI[:,1], bins=40, color=color,
    density=True, label="2", alpha = 0.5)
ax.hist(XI[:,2], bins=40, color=color,
    density=True, label="3", alpha = 0.25)
#ax.yaxis.tick_right()
plt.xlim(-3, 3)
#plt.ylim(0, 0.12)
#plt.yticks(np.arange(0, 0.121, 0.04))
plt.legend(loc="upper left")
plt.tight_layout()
fig.savefig(f"img/load_hist.pdf", 
    transparent=True, format="pdf")
plt.close()

def table_init(head, fmt):
    tab = list()
    tab.append("\\begin{table}[hpt!] \n")
    tab.append("\\begin{center} \n")
    tab.append("\\begin{tabular}{" + fmt + "} \n")
    tab.append("\\toprule \n")    
    tab.append(th + " \n")    
    tab.append("\\midrule \n")    
    return(tab)

def table_end(tab, cap = "", lab = ""):
    tab.append("\\bottomrule \n")    
    tab.append("\\end{tabular} \n")
    tab.append("\\end{center} \n")
    tab.append("\\caption{" + cap + "} \n")
    tab.append("\\label{tab:" + lab + "} \n")
    tab.append("\\end{table} \n")
    tab.append("\n \n")
    return(tab)


### Disprepancy and bias for GPS table header ###
th = "$p$ & $\\max_x \s[-1] Q(x)$"
th = th + " & $\Exp \s Q(\\hat{x}_{\\flat})$"
th = th + " & $\Exp \s \\mf_p(\\scrH_{\\flat})$"
th = th + " & $\Exp \s |\\scrE_p(\scrH_{\\flat})|$" 
th = th + " & $p \s \Exp \s |\\scrE_p(\\scrH_{\\flat})|^2$ \\\\"
tab_Ef = table_init(th, "rcrrrr")
cap_Ef = "Discrepancy and optimization bias metrics reported in Tables \\ref{tab:optbias} \\& \\ref{tab:discrep} recomputed with the vectors $\\scrH_{\\flat}$ and the corresponding covariance estimator. Sample mean estimates for expectation $\\Exp \\s X$ of column variable $X$ based on $10^5$ simulations."



### Optimimization bias table header ###
th = "$p$ & $\Exp \s |\hz|$" 
th = th + " & $\Exp \s |\\scrE_p(\\scrH)|$" 
th = th + " & $\Exp \s |\\scrE_p(\\scrH_{\\sharp})|$"
th = th + " & $p \s \Exp \s |\\scrE_p(\\scrH)|^2$"
th = th + " & $p \s |\Exp \s \\scrE_p(\\scrH_{\\sharp})|^2$ \\\\"
tab_EH = table_init(th, "rcccrr")
cap_EH = "Quadratic optimization bias length $|\\scrE_p(\\scrH)|$ for {\\pca}, its estimator $|\\hz|$ of Theorem \\ref{thm:pcabias} and $|\\scrE_p(\\scrH_{\\sharp})|$ are shown for growing $p$. Scaled variables $p \\s |\\scrE_p(\\scrH)|^2$ and $p \\s |\\scrE_p(\\scrH_{\\sharp})|^2$ are provided to illustrate the convergence rates. Sample mean estimates for expectation $\\Exp \\s X$ of column variable $X$ based on $10^5$ simulations."


### Discrepancy table header ###
th = "$p$ & $\\max_x \s[-1] Q(x)$"
th = th + " & $\\Exp \s Q(\\hat{x})$"
th = th + " & $\\Exp \s \\mf_p(\\scrH)$"
th = th + " & $\\Exp \s Q(\\hat{x}_{\\sharp})$" 
th = th + " & $\\Exp \s \\mf_p(\\scrH_{\\sharp})$ \\\\"
tab_Dp = table_init(th, "rcrrcc")
cap_Dp = "Realized maximum $Q(\\hx)$ and discrepancy $\\mf_p(\\scrH)$ of {\\pca} for growing $p$ are compared with $Q(\\hx_{\\sharp})$ and  $\\mf_p(\\scrH_{\\sharp})$, computed using the covariance $\\hSig_{\\sharp}$. The true maximum in column $2$ applies the true covariance matrix $\\bSig$. Sample mean estimates for expectation $\\Exp \\s X$ of column variable $X$ based on $10^5$ simulations."


### PCA stats table header ###
th = "$p$ & $\\Exp\s |\\scrH^\\top_{\\sharp}\\scrB\\scrB^\\top\\scrH_{\\sharp} - \\Phi^2|$ & $\\Exp \\s |\\Phi^2|$"
th = th + " & $\\Exp \\s |\\scrH^\\top \\scrB\\scrB^\\top\\scrH - \snr^2|$"
th = th + " & $\\Exp \\s |\\snr^2|$ \\\\"
tab_HB = table_init(th, "rcccc")
cap_HB = "The inner products of the columns of $\\scrH$ ({\\pca}) and $\\scrH_{\\sharp}$ after projection onto $\\col{\\scrB}$, and  their estimators $\\snr^2$ and $\\Phi^2$. The norms $|\\snr^2|$ and $|\\Phi^2|$ estimate the largest, squared projected lengths of the columns of $\\scrH$ and $\\scrH_{\\sharp}$ respectively. Sample mean estimates for expectation $\\Exp \\s X$ of column variable $X$ based on $10^5$ simulations."

### Portfolio volatility table header ###
th = "$p$ & $\\sigma_{\\text{opt}}$"
th = th + " & $\Exp\s\\tru_p(\\scrH)$"
th = th + " & $\Exp\s\\hat{\\sigma}_p(\\scrH)$"
th = th + " & $\Exp\s D_p (\\scrH)$"
th = th + " & $\Exp\s\\tru_p(\\scrH_{\\sharp})$"
th = th + " & $\Exp\s\\hat{\\sigma}_p(\\scrH_{\\sharp})$"
th = th + " & $\Exp\s D_p(\\scrH_{\\sharp})$ \\\\"


th = "$p$ & \\textsc{opt} "
th = th + " & \\textsc{pca tv}"
th = th + " & \\textsc{pca ev}"
th = th + " & $\\calP$"
th = th + " & \\textsc{jsm tv}"
th = th + " & \\textsc{jsm ev}"
th = th + " & $\\calP$ \\\\"

tab_Vp = table_init(th, "rccccccc")
cap_Vp = "Portfolio volatility statistics ($n = "
cap_Vp = cap_Vp + f"{n}"  + "$, $\\mu = "
cap_Vp = cap_Vp +  f"{mus[pidx]}$,  {num_sims} simulations)."
#cap_Vp = "Realized minimum variance portfolio volatilities (square root of $\\tru_p^2 = \\ip{\hw}{\\bSig \\hw}$) for PCA $\\scrH$ and the correction $\\scrH_\sharp$."


th = "$p$ & \\textsc{opt} "
th = th + " & \\textsc{pca tsr}"
th = th + " & \\textsc{pca esr}"
th = th + " & \\calD"
th = th + " & \\textsc{jsm tsr}"
th = th + " & \\textsc{jsm esr}"
th = th + " & \\calD \\\\"

tab_SR = table_init(th, "rccccccc")
cap_SR = "True/Estimated Sharp Ratio (SR) statistics ($n = "
cap_SR = cap_SR + f"{n}"  + "$, $\\mu = "
cap_SR = cap_SR +  f"{mus[pidx]}$,  {num_sims} simulations)."

th = "$p$ & PCA MV TR "
th = th + " & PCA CP TR"
th = th + " & PCA TR"
th = th + " & PCA MV ER"
th = th + " & PCA CP ER"
th = th + " & PCA ER \\\\"

tab_pcaRD = table_init(th, "rcccccc")
cap_pcaRD = "PCA True/Esimtated Return decompositions -- " 
cap_pcaRD = cap_pcaRD + "Minimum Variance (MV) and Characteristic Portfolio (CP)" 
cap_pcaRD = cap_pcaRD + " -- ($n = "
cap_pcaRD = cap_pcaRD + f"{n}"  + "$, $\\mu = "
cap_pcaRD = cap_pcaRD +  f"{mus[pidx]}$,  {num_sims} simulations)."

th = "$p$ & JSM MV TR "
th = th + " & JSM CP TR"
th = th + " & JSM TR"
th = th + " & JSM MV ER"
th = th + " & JSM CP ER"
th = th + " & JSM ER \\\\"


tab_mjsRD = table_init(th, "rcccccc")
cap_mjsRD = "JSM True/Esimtated Return decompositions -- " 
cap_mjsRD = cap_mjsRD + "Minimum Variance (MV) and Characteristic Portfolio (CP)" 
cap_mjsRD = cap_mjsRD + " -- ($n = "
cap_mjsRD = cap_mjsRD + f"{n}"  + "$, $\\mu = "
cap_mjsRD = cap_mjsRD +  f"{mus[pidx]}$,  {num_sims} simulations)."

th = "$p$ & \\textsc{opt} "
th = th + " & \\textsc{pca tr}"
th = th + " & \\textsc{pca er}"
th = th + " & \\calD"
th = th + " & \\textsc{jsm tr}"
th = th + " & \\textsc{jsm er}"
th = th + " & \\calD \\\\"


tab_RS = table_init(th, "rccccccc")
cap_RS = "True/estimated return statistics -- " 
cap_RS = cap_RS + "True Return (TR) and Estimated Return (ER)" 
cap_RS = cap_RS + " -- ($n = "
cap_RS = cap_RS + f"{n}"  + "$, $\\mu = "
cap_RS = cap_RS +  f"{mus[pidx]}$,  {num_sims} simulations)."


tab_ary = [tab_Vp, tab_SR, tab_RS, tab_pcaRD, tab_mjsRD] 
#, tab_EH, tab_Dp, tab_Ef, tab_HB]
Dp_pca_ary = list()
Dp_mfs_ary = list()
V2_pca_ary = list()
V2_mfs_ary = list()

for (p_sim, p_info) in zip(Z,P):
    print("\n% ---------------------")
    p = p_sim[0]['p']
    print(f"% p = {p}")
    print("% ---------------------")

    ids = range(num_sims)
    pca_bias = np.array(list(map(lambda i : p_sim[i]['pca_bias'],ids)))
    pca_bias = np.round(np.mean(pca_bias, axis=0), 4)
    print("PCA quadratic optimization bias vector.")
    print(pca_bias)
    EHpca = list(map(lambda i : norm(p_sim[i][f'pca_bias']), ids))
    EHpca = np.mean(EHpca)
    print(f"PCA bias length = {round(EHpca,3)}.\n")
    
    mfs_bias = np.array(list(map(lambda i : p_sim[i]['mfs_bias'],ids)))
    mfs_bias = np.round(np.mean(mfs_bias, axis=0), 4)
    EHmfs = list(map(lambda i : norm(p_sim[i][f'mfs_bias']), ids))
    EHmfs = np.mean(EHmfs)
    print("MFS quadratic optimization bias vector.")
    print(mfs_bias)
    print(f"MFS bias length = {round(EHmfs,3)}.\n")


    ### PCA table ###
    Phi2max = list(map(lambda i : max(p_sim[i]['Phi2']), ids))
    Phi2max = round(np.mean(Phi2max),4)
    Psi2max = list(map(lambda i : max(p_sim[i]['Psi2']), ids))
    Psi2max = round(np.mean(Psi2max),4)
        
    HBBH_Phi2 = list(map(lambda i : p_sim[i]['HBBH_sharp_Phi2'], ids))
    HBBH_Phi2 = round(np.mean(HBBH_Phi2),4)
    HBBH_Psi2 = list(map(lambda i : p_sim[i]['HBBH_Psi2'], ids))
    HBBH_Psi2 = round(np.mean(HBBH_Psi2),4)

    row = f"${p}$ & ${HBBH_Phi2}$ & ${Phi2max}$"
    row = row + f" & ${HBBH_Psi2}$ & ${Psi2max}$"
    row = row + f" \\\\ \n"
    tab_HB.append(row)

    ### Optimization bias table ###
    phil = list(map(lambda i : p_sim[i]['phil'], ids))
    phil = round(np.mean(phil),3)
        
    row = f"${p}$ & ${phil}$"
    row = row + f" & ${round(EHpca,3)}$"
    row = row + f" & ${round(EHmfs,3)}$" 
    row = row + f"& ${round(p * EHpca**2,1)}$"
    row = row + f" & ${round(p * EHmfs**2,2)}$"
    row = row + " \\\\ \n"
    tab_EH.append(row)
    

    ### Discrepancy table ###
    maxQx = round(p_info['maxQx'],2)
    Dp_pca = list(map(lambda i : p_sim[i][f'pca_Dp'], ids))
    Dp_pca_ary.append(Dp_pca)
    Dp_pca = round(np.mean(Dp_pca),2)
    Qx_pca = list(map(lambda i : p_sim[i][f'pca_Qx'], ids))
    Qx_pca = round(np.mean(Qx_pca),2)
    Dp_mfs = list(map(lambda i : p_sim[i][f'mfs_Dp'], ids))
    Dp_mfs_ary.append(Dp_mfs)
    Dp_mfs = round(np.mean(Dp_mfs),2)
    Qx_mfs = list(map(lambda i : p_sim[i][f'mfs_Qx'], ids))
    Qx_mfs = round(np.mean(Qx_mfs),2)

    row = f"${p}$ & ${round(maxQx,2)}$"
    row = row + f" & ${Qx_pca}$ & ${Dp_pca}$"
    row = row + f" & ${Qx_mfs}$ & ${Dp_mfs}$"
    row = row + " \\\\ \n"
    tab_Dp.append(row)
    
 
    ### Portfolio volatility table ###
    V2_pca = list(map(lambda i : p_sim[i][f'pca_act_pvol'], ids))
    V2_pca_ary.append(V2_pca)
    V2_pca = np.mean(np.array(V2_pca))
    V2_mfs = list(map(lambda i : p_sim[i][f'mfs_act_pvol'], ids))
    V2_mfs_ary.append(V2_mfs)
    V2_mfs = np.mean(np.array(V2_mfs))


    hV2_pca = list(map(lambda i : p_sim[i][f'pca_est_pvol'], ids))
    hV2_pca = np.mean(np.array(hV2_pca))
    hV2_mfs = list(map(lambda i : p_sim[i][f'mfs_est_pvol'], ids))
    hV2_mfs = np.mean(np.array(hV2_mfs))

    PV = p_info['pv'][pidx]
    row = f"${p}$ & ${round(PV,2)}$"
    row = row + f" & ${round(V2_pca,2)}$ & ${round(hV2_pca,2)}$" 
    row = row + f" & ${round(hV2_pca/V2_pca,2)}$" 
    row = row + f" & ${round(V2_mfs,2)}$ & ${round(hV2_mfs,2)}$" 
    row = row + f" & ${round(hV2_mfs/V2_mfs,2)}$" 
    row = row + " \\\\ \n"
    tab_Vp.append(row)

    # Sharpe Ratio table
    PV = p_info['pv']
    PR = p_info['pr']
    SR = (np.array(PR) - rf_rate) / np.array(PV)
    SR = SR[pidx]
    row = f"${p}$ & ${round(SR,2)}$"
    
    PV = list(map(lambda i : p_sim[i][f'pca_pv_act'][pidx], ids))
    PR = list(map(lambda i : p_sim[i][f'pca_pr_act'][pidx], ids))
    SR = np.mean((np.array(PR) - rf_rate)/np.array(PV))
    row = row + f" & ${round(SR,2)}$" 
    SRA = SR

    PV = list(map(lambda i : p_sim[i][f'pca_pv_est'][pidx], ids))
    PR = list(map(lambda i : p_sim[i][f'pca_pr_est'][pidx], ids))
    SR = np.mean((np.array(PR) - rf_rate)/np.array(PV))
    row = row + f" & ${round(SR,2)}$" 

    PR = list(map(lambda i : p_sim[i][f'pca_pr_act'][pidx], ids))
    PR = np.mean(np.array(PR))
    row = row + f" & ${round(SR/SRA,2)}$" 

    PV = list(map(lambda i : p_sim[i][f'mfs_pv_act'][pidx], ids))
    PR = list(map(lambda i : p_sim[i][f'mfs_pr_act'][pidx], ids))
    SR = np.mean((np.array(PR) - rf_rate)/np.array(PV))
    row = row + f" & ${round(SR,2)}$" 
    SRA = SR

    PV = list(map(lambda i : p_sim[i][f'mfs_pv_est'][pidx], ids))
    PR = list(map(lambda i : p_sim[i][f'mfs_pr_est'][pidx], ids))
    SR = np.mean((np.array(PR) - rf_rate)/np.array(PV))
    row = row + f" & ${round(SR,2)}$" 
    
    PR = list(map(lambda i : p_sim[i][f'mfs_pr_act'][pidx], ids))
    PR = np.mean(np.array(PR))
    row = row + f" & ${round(SR/SRA,2)}$" 
    
    row = row + " \\\\ \n"
    tab_SR.append(row)
    
    # Return Decomposition table
    row = f"${p}$ "
    MVTR = list(map(lambda i : p_sim[i][f'pca_mvTR'], ids))
    MVER = list(map(lambda i : p_sim[i][f'pca_mvER'], ids))
    SFTR = list(map(lambda i : p_sim[i][f'pca_sfTR'], ids))
    SFER = list(map(lambda i : p_sim[i][f'pca_sfER'], ids))
    pca_TR = np.array(MVTR) + np.array(SFTR)
    pca_ER = np.array(MVER) + np.array(SFER)
    row = row + f" & ${np.round(np.mean(MVTR),2)}$" 
    row = row + f" & ${np.round(np.mean(SFTR),2)}$" 
    row = row + f" & ${np.round(np.mean(MVTR)+np.mean(SFTR),2)}$" 
    row = row + f" & ${np.round(np.mean(MVER),2)}$" 
    row = row + f" & ${np.round(np.mean(SFER),2)}$" 
    row = row + f" & ${np.round(np.mean(MVER)+np.mean(SFER),2)}$" 
    
    row = row + " \\\\ \n"
    tab_pcaRD.append(row)

    row = f"${p}$ "
    MVTR = list(map(lambda i : p_sim[i][f'mfs_mvTR'], ids))
    MVER = list(map(lambda i : p_sim[i][f'mfs_mvER'], ids))
    SFTR = list(map(lambda i : p_sim[i][f'mfs_sfTR'], ids))
    SFER = list(map(lambda i : p_sim[i][f'mfs_sfER'], ids))
    mjs_TR = np.array(MVTR) + np.array(SFTR)
    mjs_ER = np.array(MVER) + np.array(SFER)
    row = row + f" & ${np.round(np.mean(MVTR),2)}$" 
    row = row + f" & ${np.round(np.mean(SFTR),2)}$" 
    row = row + f" & ${np.round(np.mean(MVTR)+np.mean(SFTR),2)}$" 
    row = row + f" & ${np.round(np.mean(MVER),2)}$" 
    row = row + f" & ${np.round(np.mean(SFER),2)}$" 
    row = row + f" & ${np.round(np.mean(MVER)+np.mean(SFER),2)}$" 
    
    row = row + " \\\\ \n"
    tab_mjsRD.append(row)

    PR = p_info['pr'][pidx]
    row = f"${p}$ & ${round(PR,2)}$"
    row = row + f" & ${np.round(np.mean(pca_TR),2)}$" 
    row = row + f" & ${np.round(np.mean(pca_ER),2)}$" 
    row = row + f" & ${np.round(np.mean(pca_ER/pca_TR),2)}$" 
    row = row + f" & ${np.round(np.mean(mjs_TR),2)}$" 
    row = row + f" & ${np.round(np.mean(mjs_ER),2)}$" 
    row = row + f" & ${np.round(np.mean(mjs_ER/mjs_TR),2)}$" 
    
    row = row + " \\\\ \n"
    tab_RS.append(row)



 
    flag = list(map(lambda i : p_sim[i][f'pca_minvar_flag'], ids))
    print(f"PCA minvar_flag : {sum(flag)}")
    flag = list(map(lambda i : p_sim[i][f'mfs_minvar_flag'], ids))
    print(f"MFS minvar_flag : {sum(flag)}")


    print(f"PCA volatility ratio : {round(V2_pca / hV2_pca,2)}")
    print(f"PCA volinverse ratio : {round(hV2_pca / V2_pca,2)}")
    print(f"MFS volatility ratio : {round(V2_mfs / hV2_mfs,2)}")
    print(f"MFS volinverse ratio : {round(hV2_mfs / V2_mfs,2)}")
    

   
table_end(tab_EH, cap_EH, "optbias")
table_end(tab_Dp, cap_Dp, "discrep")
table_end(tab_HB, cap_HB, "HB")
table_end(tab_Vp, cap_Vp, "vol")
table_end(tab_SR, cap_SR, "SR")
table_end(tab_RS, cap_RS, "RS")
table_end(tab_pcaRD, cap_pcaRD, "pcaRD")
table_end(tab_mjsRD, cap_mjsRD, "jsmRD")

for i in range(len(tab_ary)):
    file = open("tab/table" + str(i) + ".tex", "w")
    file.writelines(tab_ary[i])
    file.close()


# Fig 1. 
pvals = list(map(lambda i : P[i]['p'], range(len(P))))
fig, ax = plt.subplots(figsize=(9,5.56231))

Dp_ary = np.array(Dp_mfs_ary)
Dp_ave = np.mean(Dp_ary, axis = 1)
Dp_std = np.std(Dp_ary, axis = 1)
error = 2 * Dp_std
lower = Dp_ave - error
upper = Dp_ave + error
ax.plot(pvals, Dp_ave, label='$\mathscr{H}_{\\sharp}$', color = 'dimgray', 
    linewidth = 1.75, linestyle = "solid", alpha = 0.9)
ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
ax.fill_between(pvals, lower, upper, alpha=0.9, facecolor = 'gray')


if (False):
    Dp_ave = np.mean(Dp_ary, axis = 1)
    Dp_std = np.std(Dp_ary, axis = 1)
    error = 2 * Dp_std
    lower = Dp_ave - error
    upper = Dp_ave + error
    ax.plot(pvals, Dp_ave, label='$\mathscr{H}_{\\flat}$', 
        color = 'dimgray', 
        linewidth = 1.75, linestyle = "dashdot", alpha = 0.6)
    ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
    ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
    ax.fill_between(pvals, lower, upper, alpha=0.6, facecolor = 'gray')


Dp_ary = np.array(Dp_pca_ary)
Dp_ave = np.mean(Dp_ary, axis = 1)
Dp_std = np.std(Dp_ary, axis = 1)
error = 2 * Dp_std
lower = Dp_ave - error
upper = Dp_ave + error

ax.plot(pvals, Dp_ave, label='$\mathscr{H}$ ($\\textsc{pca}$)', 
    color = 'dimgray', linewidth=1.75, linestyle = "dashed", alpha = 0.3)
ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
ax.fill_between(pvals, lower, upper, alpha=0.3, facecolor = 'gray')
ax.set_xlabel('$p$ (dimension)')
ax.set_ylabel('$\hat{D}_p$ (discrepancy)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc="best")
plt.tight_layout()
plt.savefig("img/discrepancy." + "pdf",
    transparent=True, format="pdf")
 
#plt.show()
plt.close()

### Fig 2. 

fig, ax = plt.subplots(figsize=(9,5.56231))
V2_ary = np.array(V2_pca_ary)

V2_ary = np.array(V2_pca_ary)
V2_ave = np.mean(V2_ary, axis = 1)
V2_std = np.std(V2_ary, axis = 1)
error = 2 * V2_std
lower = V2_ave - error
upper = V2_ave + error

ax.plot(pvals, V2_ave, label='$\mathscr{H}$ ($\\textsc{pca}$)',  
    color = 'dimgray', linewidth=1.75, linestyle = "dashed", alpha = 0.3)
ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
ax.fill_between(pvals, lower, upper, alpha=0.3, facecolor = 'gray')
ax.set_xlabel('$p$ (portfolio size)')
ax.set_ylabel('$\mathscr{V}_p \, $ (portfolio volatility)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

V2_ave = np.mean(V2_ary, axis = 1)
V2_std = np.std(V2_ary, axis = 1)
error = 2 * V2_std
lower = V2_ave - error
upper = V2_ave + error

ax.plot(pvals, V2_ave, label='$\mathscr{H}_{\\flat}$', 
    color = 'dimgray', linewidth=1.75, linestyle = "dashdot", 
    alpha = 0.6)
ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
ax.fill_between(pvals, lower, upper, alpha=0.6, facecolor = 'gray')

V2_ary = np.array(V2_mfs_ary)
V2_ave = np.mean(V2_ary, axis = 1)
V2_std = np.std(V2_ary, axis = 1)
error = 2 * V2_std
lower = V2_ave - error
upper = V2_ave + error

ax.plot(pvals, V2_ave, label='$\mathscr{H}_{\\sharp}$', 
    color = 'dimgray', linewidth=1.75, alpha = 0.9)
ax.plot(pvals, lower, color='gray', alpha=0.0, linewidth=0)
ax.plot(pvals, upper, color='gray', alpha=0.0, linewidth=0)
ax.fill_between(pvals, lower, upper, alpha=0.9, facecolor = 'gray')

ax.legend(loc = "best")

plt.tight_layout()
fig.savefig(f"img/volatility.pdf", 
    transparent=True, format="pdf")
plt.close()




