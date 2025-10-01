from multiprocessing.pool import Pool
from multiprocessing import Lock
from threading import current_thread
import numpy as np
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
import pickle
import cvxpy as cvp

eps = 9e-15  # machine tolerance

#pvals = np.arange(1000, 8001, 1000)
pvals = [500, 1000, 2000, 3000, 100000]
pmax = max(pvals) # max number of variables


n = 125 # number of observations
nthreads = 16
num_sims = 100

# random number generator
mod_seed = 0 # model seed
sim_seed = 0 # simulation seed
mrng = np.random.default_rng(mod_seed)

norm = lambda x : np.linalg.norm(x, ord=2)
sqrt = np.sqrt


def ortho(M):  
    vals, vecs = eigh(M.T @ M)
    V = vals[::-1] 
    sM = M @ np.fliplr(vecs) / sqrt(V)

    return(sM)


# project U onto col(M)
def proj(U, M):
    sM = ortho(M)
    return sM @ (sM.T @ U)

###################
### Model Setup ###
###################

num_style = 2
num_block = 4
# total number of factors
q = 1 + num_style + num_block

fvol = np.array([16, 4, 2, 20, 15, 10, 5])
C = np.array([
       [ 1.00,  0.28, -0.30,  0.16,  0.08, 0.04,  0.02],
       [ 0.28,  1.00, -0.11,  0.00,  0.00, 0.00,  0.00],
       [-0.30, -0.11,  1.00,  0.00,  0.00, 0.00,  0.00],
       [ 0.16,  0.00,  0.00,  1.00,  0.00, 0.00,  0.00],
       [ 0.08,  0.00,  0.00,  0.00,  1.00, 0.00,  0.00],
       [ 0.04,  0.00,  0.00,  0.00,  0.00, 1.00,  0.00],
       [ 0.02,  0.00,  0.00,  0.00,  0.00, 0.00, 1.00]])

# symmetrize just in case --
C = (C + C.T) / 2

F = np.outer(fvol, fvol) * C
vals, vecs = eigh(F)
A = vecs @ np.diag(sqrt(vals))

mus = list(np.arange(7,14 + eps, 0.5))
ees = list(np.ones(len(mus)))

if (max(np.shape(F)) == np.linalg.matrix_rank(F) == q):
    print("Factor variance matrix okay!")
else:
    print("Factor error!")
    exit()

#fvol = sqrt(np.diag(F))
#C = F * np.outer(1/fvol, 1/fvol)
print(f"corr(f) = \n {np.round(C,2)}")
print(f"\n var(f) = \n {np.round(F,2)}")


## Factor exposures

# -- market factor
mu = 1
sg = 0.25
beta = mrng.normal(mu, sg, pmax)

# -- style factors
BS = np.zeros((pmax, num_style))
for k in range(num_style):
    zeta = mrng.normal(0, 1 / (k + 1), pmax)
    #zeta = zeta  - np.mean(zeta)
    #zeta = zeta / np.sqrt(np.var(zeta))
    BS[:, k] = zeta # mean 0 and sdev 1

# -- block factors (country, industry, etc)

#prob = 1 / np.diag(F)[-num_block:] 
#prob = prob / sum(prob)
BB = np.zeros((pmax, num_block))
for i in range(pmax):
    block_ids = list(range(num_block))
    k1, k2 = mrng.choice(block_ids, 2) #, p=prob)
    BB[i,k1] = mrng.uniform(0,1)
    BB[i,k2] = BB[i, k2] + mrng.uniform(0,1)

XI_block = np.copy(BB)

# specific volatilities
svol = 15.0 + mrng.beta(4, 16, pmax) * 100
svar = svol**2

# Assemble factor model 
B = np.hstack((BS, BB))
B = np.insert(B, 0, beta, axis=1)

XI = np.copy(B)
# use exposures to make return vector
fret = 12.0 * np.array([0.5, 0.29, 0.37, 0, 0, 0, 0])
sret = 0.5 / 100 * (svar - proj(svar, B))
Bret = XI @ fret 
alpha = Bret + sret
print(f"Mean expected return = {np.mean(alpha)}")
print(f"Mean specific return = {np.mean(sret)}")
print(f"Mean factor return = {np.mean(Bret)}")

pones = np.ones(pmax)

# Final loadings matrix
B = B @ A
Lams = np.linalg.eigvalsh( B.T @ B) / pmax
print(f"Lamdba matrix = \n {np.round(Lams,1)} \n")

Iq = np.eye(q)
In = np.eye(n)
qones = np.ones(q)
nones = np.ones(n)

Pn = In - np.outer(nones, nones) / n
#Pn = In

paths = mrng.choice(num_sims, 1)
pidx = 3
if (pidx > len(mus)):
    pidx = len(mus) - 1

_RUN_ = True #False

if (norm(Pn - In) > eps):
    print("Data centering is turned on!")


############################################
# Tools, Functions                         #
############################################

def markowitz_osqp(B, fv, sv, m, mu1 = 1.0, mu2 = 1.0):
    
    p = len(svar)
    Sigma = (B * fv) @ B.T + np.diag(sv)
    x = cvp.Variable(p)
    risk = cvp.quad_form(x,Sigma)
    ret = cvp.Problem(cvp.Minimize(risk),
                     [cvp.sum(x) == mu2, x @ m >= mu1])
    ret.solve(solver=cvp.OSQP,
        eps_abs=9E-12, 
        eps_rel=0, 
        verbose=False, 
        max_iter=int(10000000))
    
    return x.value


def M_abcd(H, sv, m, e):

    HTsv = H.T / sv
    esvar = e.T / sv
    msvar = m.T / sv
    HTsv_e = HTsv @ e
    HTsv_m = HTsv @ m

    # A = Iq + HTsv @ H and M = inv(A)
    # solve A x = HTsv_m to get M @ Hsv_m
    A = Iq + HTsv @ H
    MHTsv_e = np.linalg.solve(A, HTsv_e)
    MHTsv_m = np.linalg.solve(A, HTsv_m)
    
    a = esvar @ m - HTsv_e @ MHTsv_m
    b = msvar @ m - HTsv_m @ MHTsv_m
    c = esvar @ e - HTsv_e @ MHTsv_e
    d = b*c - a**2 
     
    return(MHTsv_e, MHTsv_m, a, b, c, d)


def M_zeta(H, sv, m, e, mu1, mu2):

    MHTsv_e, MHTsv_m, a, b, c, d = M_abcd(H, sv, m, e)
    
    Hsv = H.T / sv
     
    phi = mu2 * b - mu1 * a
    psi = mu1 * c - mu2 * a
    zeta = (phi * e + psi * m) / d
    
    M_zeta_e = phi * Hsv.T @ MHTsv_e / d
    M_zeta_m = psi * Hsv.T @ MHTsv_m / d

    zeta_e = phi * e / d
    zeta_m = psi * m / d

    return(zeta_e, zeta_m, M_zeta_e, M_zeta_m)


def markowitz(H, fv, sv, m, mu1 = 1.0, mu2 = 1.0):
    
    p, q = np.shape(H)
    e = np.ones(p)
 
    if not isinstance(mu1, list):
        mu1 = list([mu1])
        mu2 = list([mu2])
    
    # fix me -- remove fv everywhere!!!
    H = H * sqrt(fv)

    w_list = list()
    mus = list(zip(mu1,mu2))
    
    for mu in mus:
        mu1, mu2 = mu
     
        w = minvar(H, qones, sv, mu2)
        if (w @ m < mu1): # minvar portfolio returns less
            z_e, z_m, Mz_e, Mz_m = M_zeta(H, sv, m, e, mu1, mu2)
            
            w_e = z_e / sv - Mz_e
            w_m = z_m / sv - Mz_m
            w = w_e + w_m
            
        w_list.append(w)
    
    if len(w_list) > 1:
        return (w_list)
    else:
        return(w_list.pop())

def bullet(w_list, m, H, fv, sv):
    
    pv = list() # portfilio vol
    pr = list() # portfolio return
    for w in w_list:
        sys_risk = (H * sqrt(fv)).T @ w
        sys_risk = sys_risk @ sys_risk
        #print(f"sys = {sys_risk}")
        #print(f"spc = {(w * svar) @ w}")
        pv.append(sqrt(sys_risk + (w * sv) @ w))
        pr.append(w @ m)

    return(pv, pr)


def markowitz_check(H, fv, sv, m, mu1, mu2):
    
    x = markowitz_osqp(H, fv, sv, m, mu1, mu2)
    w = markowitz(H, fv, sv, m, mu1, mu2)
    return(norm(w-x))


def minvar_(B, fact_vars, spec_vars):
    M = B.T / spec_vars
    A = np.diag(1.0 / fact_vars) + M @ B
    b = np.sum(M, 1)
    theta = np.linalg.solve(A,b)
    w = (1.0 - B @ theta) / spec_vars
    return(w)


def minvar(B, fact_vars, spec_vars, mu2 = 1.0):
    w = minvar_(B, fact_vars, spec_vars)
    return (mu2 * w / np.sum(w))


############################################
# Sample function                          #
############################################

def sample(trial_num):
    X = rng.normal(0.0, qones, (n, q))
    Z = rng.normal(0.0, svol * np.ones(pmax), (n, pmax)).T
    

    Y = (np.outer(alpha, nones) + B @ X.T + Z) 
    m = Y @ nones / n
    Y = Y @ Pn # center


    res = list()
    for p in pvals:
        res_ = compute(Y[0:p,:], X, B[0:p,:], svar[0:p], m[0:p])
        res.append(res_)
    
    return(res)


def compute(Y, X, B, svar, m):
    res = dict()
    
    sB = ortho(B)
    p, n = Y.shape
    e = np.ones(p)
    
    # Marchenko-Pastur correction param
    npc = n / p
    
    # adjust n in case of centered data
    if (norm(Pn - In) > eps):
        n = n - 1
        # todo : otherwise what do we do with m?

    # out of sample return
    x_ = rng.normal(0.0, qones, (1, q))
    z_ = rng.normal(0.0, svol[0:p] * np.ones(p), (1, p)).T
    y_ = np.reshape(alpha[0:p],(p,1)) + B @ x_.T + z_ 


    ### PCA calculations ###
    L = Y.T @ Y / p   # dual covariance matrix
    vals, vecs = eigsh(L, q, which='LA')
    vals = vals[::-1] 
    vecs = np.fliplr(vecs)
    sH = Y @ vecs / np.sqrt(p * vals)  
    eigs = vals * p / n
 

    #gam2 = (np.trace(L) - sum(vals))*(1 + npc) / (n-q-npc) 
    gam2 = (np.trace(L) - sum(vals)) / (n-q-npc*q) 
    Psi2 = (vals - gam2) / vals 
    fvar_est = Psi2 * vals * p / n 
    fvol_est = np.sqrt(fvar_est)

    BTBp = B.T @ B / p
    Lam_pn, vecs_ = eigsh(X @ BTBp @ X.T / n, q, which='LA')
    Lam_pn = np.flip(Lam_pn)

    #D1 = np.diag(Y @ Y.T / n - sH * eigs @ sH.T)
    #D2 = np.sum((Y / sqrt(n) - (sH * sqrt(eigs)) @ vecs.T)**2, axis = 1) 
    #print(norm(D1 -  D2)) # pass on zero!!!

    Delta = np.sum((Y / sqrt(n) - (sH*sqrt(eigs)) @ vecs.T)**2, axis = 1) 
    #print(f"Dpca  sum = {np.sum(Delta)}")
    #print(f"Trace chk = {np.sum(np.diag(Y @ Y.T / n)) -np.sum(eigs)}")
    #print(f"Dpca mean = {n / p * np.sum(Delta) / (n - q) }")
    #print(f"bulk mean = {(np.trace(L)-sum(vals)) / (n-q)}")
    #print(f"svar mean = {np.mean(svar)}")
    #print(f"gam2 = {gam2}")
    svar_est = Delta

    # enforce the relation gam2 = mean(Delta)
    Delta = Delta * gam2 / np.mean(Delta)
    #print(np.mean(Delta))


    ### shrinkage vector ###
    z = e / norm(e)  #
    ########################

    # JS for the means
    gm = (m @ e) * e / p
    m2 = np.sum((m - gm)**2)
    c = 1 - (p * gam2 / n) / m2
    mjs = c * m + (1 - c) * gm 
    
    sHTz = sH.T @ z
    zH = sH @ sHTz
    zzH = sqrt(1 - z @ zH) # = |z - zH|
    zpH = (z - zH) / zzH

    Psi = np.sqrt(Psi2)
    Pi = ( 1 - Psi2 ) / Psi
    phi = Pi * sHTz.T / zzH

    pca_bias = sB.T @ zpH
    
    sHsB = sH.T @ sB 
    HBBH = sHsB @ sHsB.T
     

    wmin = minvar(sH, fvar_est, svar_est)
    wl = markowitz(sH, fvar_est, svar_est, m, mus, ees)
    w = wl[pidx]
    
    pca_minvar_flag = 0
    if ( (wmin @ m) >= (w @ m) ):
        #print(f"p = {p}")
        #print(f"HMM PCA est. return = {w @ m} vs minvar {wmin @ m}")
        pca_minvar_flag = 1

    pca_minvar_flag = 0
    if ( (wmin @ m) >= mus[pidx] ):
        print(f"p = {p} : PCA.est.ret. = {w @ m} = minvar {wmin @ m}")
        print(f"PCA.true.ret. = {w @ alpha[0:p]} vs minvar {wmin @ alpha[0:p]}")
        pca_minvar_flag = 1
        TR_e = w @ alpha[0:p]
        ER_e = w @ m
        TR_f = 0.0
        ER_f = 0.0
    else:
        H_ = sH * sqrt(fvar_est)
        mu1_ = mus[pidx]
        mu2_ = ees[pidx]
        z_e, z_m, Mz_e, Mz_m = M_zeta(H_, svar_est, m, e, mu1_, mu2_)
            
        w_e = z_e / svar_est - Mz_e
        w_f = z_m / svar_est - Mz_m
        
        TR_e = w_e @ alpha[0:p]
        ER_e = w_e @ m
        TR_f = w_f @ alpha[0:p]
        ER_f = w_f @ m
 
        #print(f"p = {p} : w dot mjs = {w @ mjs}")
        #print(f"est w_e.ret = {w_e @ mjs}")
        #print(f"tru w_e.ret = {w_e @ a}")
        #print(f"est w_m.ret = {w_m @ mjs}")
        #print(f"tru w_m.ret = {w_m @ a}")
        w_ = w_e + w_f
        diff = norm(w_-w)
        if (diff > eps):
            print("***********************************")
            print(f"****** PCA error = {diff} ********")
            print("***********************************")

    res.update(pca_out_ret = w @ y_)
    
    res.update(pca_mvTR = TR_e)
    res.update(pca_sfTR = TR_f)
    res.update(pca_mvER = ER_e)
    res.update(pca_sfER = ER_f)

    pv_est, pr_est = bullet(wl, m, sH, fvar_est, svar_est)
    pv_act, pr_act = bullet(wl, alpha[0:p], B, qones, svar)
    pca_est_pvol = np.sum(((sH.T @ w) * fvol_est)**2)
    pca_est_pvol = np.sqrt(pca_est_pvol + (w @ w) * gam2)
    pca_act_pvol = np.sum((B.T @ w)**2)
    truRisk2 = pca_act_pvol + w @ (svar * w)
    pca_act_pvol = np.sqrt(truRisk2)
     
    muhp = p / gam2
    Cpz = gam2 * (sHTz.T / (gam2 + eigs * Psi2)) @ sHTz
    muhp = muhp * (zzH**2 + Cpz)
    pca_Dp = 2 - muhp * truRisk2

    res.update(pca_pr_est = pr_est)
    res.update(pca_pv_est = pv_est)
    res.update(pca_pr_act = pr_act)
    res.update(pca_pv_act = pv_act)
    res.update(pca_Dp = pca_Dp)
    res.update(pca_Qx = 1 + (muhp / 2.0) * pca_Dp)
    res.update(pca_est_pvol=pca_est_pvol)
    res.update(pca_act_pvol=pca_act_pvol)
    res.update(pca_Dp = 2 - muhp * truRisk2)
    res.update(svar_ave = np.mean(svar))
    res.update(svar_est = gam2)
    res.update(fvar_est = fvar_est)
    res.update(Lambda = Lam_pn) # remove
    res.update(Lam_pn = Lam_pn)
    res.update(pca_bias = pca_bias)
    res.update(Psi2 = Psi2)
    res.update(HBBH_Psi2 = norm(HBBH - np.diag(Psi2)))
    res.update(phil = norm(phi))
    res.update(Hz_HBBz = norm(sH.T @ z - sHsB @ (sB.T @ z)))
    res.update(HBBH_psi2 = norm(HBBH - np.diag(Psi2))) # remove

    ### tru vectors calculations ##
    wl = markowitz(sB, fvar_est, svar_est, m, mus, ees)
    w = wl[pidx]
    tru_est_pvol = np.sum(((sB.T @ w) * fvol_est)**2)
    tru_est_pvol = np.sqrt(tru_est_pvol + (w @ w) * gam2)
    tru_act_pvol = np.sum((B.T @ w)**2)
    truRisk2 = tru_act_pvol + w @ (svar * w)
    tru_act_pvol = np.sqrt(truRisk2)

    res.update(pca_minvar_flag = pca_minvar_flag)
    res.update(tru_est_pvol = tru_est_pvol)
    res.update(tru_act_pvol = tru_act_pvol)


    ### H SHARP calculations ###
    sHz = np.vstack((sH.T, zpH)).T
    H_plus = sH * Psi + np.outer(zpH, phi)
    sH_sharp = ortho(H_plus)

    sHz = np.vstack((sH.T, zpH)).T

    ### some JS calculations ###
    T = np.reshape(z, (p,1)) # can be p x m matrix (m = 1 here)
    # D is the inverse of Iq - np.outer(sHTz, sHTz))
    D = Iq + np.outer(sHTz, sHTz) / (1 - sHTz @ sHTz)
    C = Iq - D @ (Iq - np.diag(Psi2))
    M = proj(sH, T)
    H_jse = sH @ C + M @ (Iq - C) 
    sH_jse = ortho(H_jse / Psi)
    # test JS and sharp formulas | pass on zero!
    # print(norm(Iq - sH_jse.T @ sH_sharp))


    # Unweighted JSM calculations

    sM = np.vstack((e, mjs)).T
    sM = ortho(sM) # p x 2 matrix of orthonormal columns
    N = np.linalg.inv(Iq - sH.T @ sM @ sM.T @ sH)   
    C = Iq - N @ (Iq - np.diag(Psi2))
    M = sM @ (sM.T @ sH)
    # A = np.vstack((e, m)).T
    # M_ = A @ np.linalg.inv(A.T @ A) @ A.T @ sH
    # print(norm(M - M_))
    sH_jsqp = sH @ C + M @ (Iq - C)
    sH_jsqp = ortho(sH_jsqp)

    # 2nd version
    A = np.vstack((e, m)).T
    H = sH * sqrt(eigs)
    M = (A @ np.linalg.inv(A.T @ A)) @ (A.T @ H)
    N = (H - M).T @ (H - M)
    C = Iq - np.linalg.inv(N) * (gam2 * p / n)
    sH_jsqp_ = (H @ C + M @ (Iq - C)) / sqrt(eigs) 
    sH_jsqp_ = ortho(sH_jsqp_)
    # confirm both versions give same answer | zero pass
    #print(norm(Iq - sH_jsqp_.T @ sH_jsqp))


    ### Weighted JSM calculations ###
    
    D = 1.0 / sqrt(Delta)
    YD = (Y.T * D).T
    LD = YD.T @ YD / p   # dual covariance matrix
    valsD, vecsD = eigsh(LD, q, which='LA')
    valsD = valsD[::-1] 
    vecsD = np.fliplr(vecsD)
    sHD = YD @ vecsD / np.sqrt(p * valsD)  
    eigsD = valsD * p / n
 
    gam2D = (np.trace(LD) - sum(valsD)) / (n-q-npc) 
    Psi2D = (valsD - gam2D) / valsD 

    sMD = np.vstack((e * D, mjs * D)).T
    sMD = ortho(sMD) # p x 2 matrix of orthonormal columns
    ND = np.linalg.inv(Iq - sHD.T @ sMD @ sMD.T @ sHD)   
    CD = Iq - ND @ (Iq - np.diag(Psi2D))
    MD = sMD @ (sMD.T @ sHD)
    sHD_jsqp = sHD @ CD + MD @ (Iq - CD) 
    sHD_jsqp = ortho( (sHD_jsqp.T / D).T)

    # alternative version
    AD = np.vstack((e * D, mjs * D)).T
    HD = sHD * sqrt(eigsD)
    MD = (AD @ np.linalg.inv(AD.T @ AD)) @ (AD.T @ HD)
    ND = (HD - MD).T @ (HD - MD)
    CD = Iq - np.linalg.inv(ND) * (gam2D * p / n)
    sHD_jsqp_ = (HD @ CD + MD @ (Iq - CD)) / sqrt(eigsD) # * Psi2)
    sHD_jsqp_ = ortho( (sHD_jsqp_.T / D).T )
    # confirm both versions give same answer | zero pass
    # print(norm(Iq - sHD_jsqp_.T @ sHD_jsqp))
    
    sH_jsqp = sHD_jsqp_

    # Oracle
    sHTM = sH.T @ sM
    MpH = sM - sH @ sHTM 
    MpH = ortho(MpH) # @ np.linalg.inv(MpH.T @ sM) 

    sHM = np.hstack((sH, MpH))

    T_star = sHM.T @ sB
    sH_star = ortho(sHM @ T_star * eigs)
    # careful : must use ortho for the next computation
    zmH_star = sM - sH_star @ (sH_star.T @ sM)
    E_star = sB.T @ zmH_star
    #print(f"{E_star}\n")
    
    # project z onto sH_sharp (careful : ortho must be used)
    zH_sharp = sH_sharp @ (sH_sharp.T @ z)
    mfs_bias = sB.T @ (z - zH_sharp) / sqrt(1 - z @ zH_sharp)

    
    # compute portfolio volatilites
    #sH_jsqp = sH_star
    wmin = minvar(sH_jsqp, fvar_est, svar_est, ees[pidx])

    wl = markowitz(sH_jsqp, fvar_est, svar_est, mjs, mus, ees)
    pv_est, pr_est = bullet(wl, mjs, sH_jsqp, fvar_est, svar_est)
    pv_act, pr_act = bullet(wl, alpha[0:p], B, qones, svar)
    w = wl[pidx]
    

    mfs_minvar_flag = 0
    if ( (wmin @ mjs) >= mus[pidx] ):
        print(f"p = {p} : MFS.est.ret. = {w @ mjs} = minvar {wmin @ mjs}")
        print(f"MFS.true.ret. = {w @ alpha[0:p]} vs minvar {wmin @ alpha[0:p]}")
        mfs_minvar_flag = 1
        TR_e = w @ alpha[0:p]
        ER_e = w @ mjs
        TR_f = 0.0
        ER_f = 0.0
    else:
        H_ = sH_jsqp * sqrt(fvar_est)
        mu1_ = mus[pidx]
        mu2_ = ees[pidx]
        z_e, z_m, Mz_e, Mz_m = M_zeta(H_, svar_est, mjs, e, mu1_, mu2_)
            
        w_e = z_e / svar_est - Mz_e
        w_f = z_m / svar_est - Mz_m
    
        TR_e = w_e @ alpha[0:p]
        ER_e = w_e @ mjs
        TR_f = w_f @ alpha[0:p]
        ER_f = w_f @ mjs
 
        #print(f"p = {p} : w dot mjs = {w @ mjs}")
        #print(f"est w_e.ret = {w_e @ mjs}")
        #print(f"tru w_e.ret = {w_e @ a}")
        #print(f"est w_m.ret = {w_m @ mjs}")
        #print(f"tru w_m.ret = {w_m @ a}")
        w_ = w_e + w_f
        diff = norm(w_-w)
        if (diff > eps):
            print("***********************************")
            print("********** MFS error **************")
            print("***********************************")

    res.update(mfs_out_ret = w @ y_)
   
    res.update(mfs_mvTR = TR_e)
    res.update(mfs_sfTR = TR_f)
    res.update(mfs_mvER = ER_e)
    res.update(mfs_sfER = ER_f)

    mfs_est_pvol = np.sum(((sH_jsqp.T @ w) * fvol_est)**2)
    mfs_est_pvol = np.sqrt(mfs_est_pvol + (w @ w) * gam2)
    mfs_act_pvol = np.sum((B.T @ w)**2)
    truRisk2 = mfs_act_pvol + w @ (svar * w)
    mfs_act_pvol = np.sqrt(truRisk2)

    muhp = p / gam2
    sH_shz = sH_sharp.T @ z
    Cpz = gam2 * (sH_shz.T / (gam2 + eigs * Psi2)) @ sH_shz
    muhp = muhp * ((1 - z @ zH_sharp) + Cpz)
    mfs_Dp = 2 - muhp * truRisk2

    sHsB = sH_sharp.T @ sB 
    HBBH = sHsB @ sHsB.T
    Phi2, _ = eigh(np.diag(Psi2) + np.outer(phi, phi))
    Phi2 = Phi2[::-1] 
    HBBH_Phi2 = norm(HBBH - np.diag(Phi2))

    res.update(mfs_minvar_flag = mfs_minvar_flag)
    res.update(mfs_pr_est = pr_est)
    res.update(mfs_pv_est = pv_est)
    res.update(mfs_pr_act = pr_act)
    res.update(mfs_pv_act = pv_act)
    res.update(mfs_Dp = mfs_Dp)
    res.update(mfs_Qx = 1 + (muhp / 2.0) * mfs_Dp)
    res.update(mfs_bias = mfs_bias)
    res.update(mfs_est_pvol = mfs_est_pvol)
    res.update(mfs_act_pvol = mfs_act_pvol)
    res.update(Phi2 = Phi2)
    res.update(HBBH_sharp_Phi2 = HBBH_Phi2)

    res.update(p=p)

    return(res)

# SIMULATION

def init_worker(lock):
    # get the current thread
    thread = current_thread()
    # report the name of the current thread
    with lock:
        seeds = pickle.load(open("seeds.p", "rb"))
        s = seeds.pop()
        info = f"Initializing thread {thread.native_id} :"
        info = info + f" seed entropy {s.entropy}"
        info = info + f" and spawn_key {s.spawn_key}."
        print(info, flush=True)
        
        global rng 
        rng = np.random.default_rng(s)
        pickle.dump(seeds, open("seeds.p", "wb"))
 
def main(main_seed = sim_seed):
    trials = range(1, 1 + num_sims)
    X = zip(trials)  # have to pass something

    seeds = np.random.SeedSequence(main_seed)
    child_seeds = seeds.spawn(nthreads)
    pickle.dump(child_seeds, open("seeds.p", "wb"))
    
    lock = Lock()
    pool = Pool(initializer=init_worker, 
        initargs = (lock,),
        processes = nthreads)
    
    res = pool.starmap (sample, X)
    pool.close()
    pool.join()
        
    return (list(zip (*res)))

# PLOT
if __name__ == '__main__':

    sB = ortho(B)
    z = np.ones(pmax) / sqrt(pmax)
    cond = norm(sB @ (sB.T @ z) -z)
    print(f"|z_B - z| = {cond}")
    print(f"|z_B| = {norm(sB @ (sB.T @ z))}")
    if (cond < 0.05):
        print("Warning! z too close to col(B)")
        exit()

    P = list()
    for p in pvals:
        
        Bp = B[0:p,:]
        svarp = svar[0:p]
        alphp = alpha[0:p]

        sB = ortho(Bp)
        pones = np.ones(p)
        z = pones / sqrt(p)
        cond = norm(sB @ (sB.T @ z) -z)
        print(f"p = {p} : |z_B - z| = {cond}")

        w = minvar(Bp, qones, svarp)
        opt_vol = sum((Bp.T @ w)**2) + w @ (svarp * w)
        opt_vol = np.sqrt(opt_vol)
        
        mu2p = pones @ minvar_(Bp, qones, svarp)

        p_info = dict(p=p)
        p_info.update(minvol=opt_vol)
        p_info.update(maxQx = 1 + mu2p / 2.0)
        
        wl = markowitz(Bp, qones, svarp, alphp, mus, ees)
        pv, pr = bullet(wl, alphp, Bp, qones, svarp)
        p_info.update(pv = pv)
        p_info.update(pr = pr)
        
        if (p <= 3000):
            p_info.update(XI_block = XI_block[0:p,:])
        
        P.append(p_info)

    if (_RUN_):
        Z = main()
        data = dict()
        data.update(Z=Z, P=P, num_sims=num_sims, num_block=num_block, 
            alpha = alpha, Bvol = sqrt(np.sum(B**2, axis=1)),
            svol=svol, Bret = Bret, fret = fret, sret = sret, 
            F = F, XI=XI, pmax = pmax, q=q, n=n, 
            mus = mus, pidx=pidx, paths=paths)

        pickle.dump(data, open("_data_jsm.p", "wb"))
       
        print(f"Number investment dates = {num_sims}")

        for i in range(len(pvals)):
            mfs_ret = np.zeros(num_sims)
            pca_ret = np.zeros(num_sims)
            for j in range(num_sims):
                mfs_ret[j] = Z[i][j]['mfs_out_ret']
                pca_ret[j] = Z[i][j]['pca_out_ret']
 
            print(f"p = {pvals[i]}")
            print(f"pca vol = {sqrt(np.var(pca_ret))}")
            print(f"mfs vol = {sqrt(np.var(mfs_ret))}")


   

