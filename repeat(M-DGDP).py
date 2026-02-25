# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:15:32 2023

@author: ddl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:35:25 2023

@author: ddl
"""

# --------------------------------需要导入的包-----------------------------------

from scipy import linalg  # 用于实现K-R乘积
import tensorflow as tf
import copy
import random
import math
import torch
from tensorly.decomposition import parafac
from tensorly import unfold as tl_unfold
import tensorly as tl
from sklearn.decomposition import FactorAnalysis, PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#import tensortools as tt
#from tensortools.operations import unfold as tt_unfold, khatri_rao

# torch.sum(torch.mul(W,x)) 表示W和x的内积
#random.seed(5408)
from scipy.stats import norm
from scipy.stats import invgamma  #从逆伽马分布中抽样
from scipy.stats import invgauss  #从逆高斯分布中抽样
from scipy.stats import binom  #从两点分布中抽样
from scipy.stats import geninvgauss  #从广义逆高斯分布中抽样

from scipy.stats import beta  #从贝塔分布中抽样

from scipy.stats import multivariate_normal

from scipy.stats import dirichlet

from scipy.stats import gamma
#---------------------------------调用GPU执行代码-------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

# -------------------------------需要用到的函数----------------------------------

def vec(X):  # 将矩阵拉直为列向量的函数(按列拉直)
    vec_X = list()
    for r in range(len(X.T)):
        vec_X = vec_X + list(np.array(X[:, r]))
    return vec_X

def X_d(X,d): #张量的模d折叠为矩阵(行为p_d)
    X = list(np.array(X));
    p_3 = len(X);p_2 = len(X[0].T);p_1 = len(X[0]);
    
    x_d = list();
    if d == 0:
        for i_1 in range(p_1):
            x_d_1 = list();
            for i_3 in range(p_3):
                x_d_1 = x_d_1 + list(X[i_3][i_1,:]);
            x_d.append( np.array(x_d_1) );
            
    if d == 1:
        for i_2 in range(p_2):
            x_d_2 = list();
            for i_3 in range(p_3):
                x_d_2 = x_d_2 + list(X[i_3][:,i_2]);
            x_d.append( np.array(x_d_2) );
            
    if d == 2:
        for i_3 in range(p_3):
            x_d_3 = vec(X[i_3]);
            x_d.append( np.array(x_d_3) );
                
    return np.matrix(x_d)



def griddy(hua_A,MM,iteration_a,iteration_W,R,D,g,pd):#网络格点抽样算法

    
    bb_delta_r = list();
    
    for r in range(R):
        bb_delta_rr = 0;
        for d in range(D):
            bb_delta_rr = bb_delta_rr + iteration_a[d][:,r].T @ np.linalg.inv(np.diag( iteration_W[d][r] )) @ iteration_a[d][:,r];
            
        bb_delta_r.append( bb_delta_rr );
        
    prod = list();
    
    for a in hua_A:
        
        delta_aa = R * a;
        
        delta_bb = a * ( R/g )**(1/D);
        
        aa_delta_r = 2*delta_bb;
        
        aa_delta = 2*delta_bb;
        
        w_alpha = list();
        
        for m in range(MM):
            
            delta_rr = list();
            
            for r in range(R):
                
                delta_rm = float(geninvgauss.rvs(p = a - 0.5*sum(pd),b = (aa_delta_r * bb_delta_r[r])**0.5, scale = (bb_delta_r[r]/aa_delta_r)**0.5,  size = 1));
                
                delta_rr.append(delta_rm);
            
            phi_m = list();
            
            sum_delta_rr = sum( delta_rr );
            
            for r in range(R):
                phi_rm = delta_rr[r]/sum_delta_rr;
                
                phi_m.append( phi_rm );
                
            bb_delta = 0;
            for r in range(R):
                bb_delta = bb_delta + bb_delta_r[r]/phi_m[r];
            
            delta_m = float(geninvgauss.rvs(p = delta_aa - 0.5*R*sum(pd),b = (aa_delta * bb_delta)**0.5, scale = (bb_delta/aa_delta)**0.5,  size = 1))
            
            w_alpha_m = 0;
  
            for r in range(R):
                
                for d in range(D):
                    
                    w_alpha_m = w_alpha_m + math.log(multivariate_normal.pdf( iteration_a[d][:,r],mean = np.zeros((pd[d])),cov = phi_m[r]*delta_m * np.diag( iteration_W[d][r] )));

            w_alpha_m = w_alpha_m + math.log(dirichlet.pdf( phi_m,a*np.ones((R)) )) + math.log(gamma.pdf( delta_m,a = delta_aa,scale = 1/delta_bb )) ;
            
            w_alpha.append( w_alpha_m );
            
        lmax = max(w_alpha);
        
        a_poserior = np.mean(list((map(lambda x:math.exp(x - lmax),w_alpha))));
        
        prod.append(a_poserior);
    
    log = logsum(prod);
    
    for t in range(len(prod)):
        prod[t] = math.exp( prod[t] - log );
        
    alpha = random.choices(hua_A, prod);
            
    return alpha


def griddy_nu(r,phi,delta,R,D,length_nu,nu_grid,iteration_a,pd):#网格子确定超参数
    
    prod = list();
    for t in range(length_nu**2):
        a_nu_r = nu_grid[t,0];b_nu_r = nu_grid[t,0]*nu_grid[t,1];
    
        poserior = 0;
        for d in range(D):
            nu_bb = sum(abs(iteration_a[d][:,r]))/(phi[r]*delta)**0.5;
            poserior = poserior + (  math.log( math.gamma(a_nu_r + pd[d]) * b_nu_r**a_nu_r )-math.log( math.gamma(a_nu_r)* (b_nu_r + nu_bb)**(a_nu_r + pd[d]) )  );
        #print(poserior)
        prod.append(poserior)
    
    log = logsum(prod);
    
    #sum_prod = sum(prod)
    
    for t in range(length_nu**2):
        prod[t] = math.exp( prod[t] - log );
        
    sample_index = random.choices(range(length_nu**2), prod);
    
    a_nu = nu_grid[sample_index,0];
    
    b_nu = nu_grid[sample_index,0] * nu_grid[sample_index,1];
    
    return a_nu,b_nu


def logsum(lx):
    ss = 0;
    for x in lx:
        ss = ss + math.exp(x - max(lx));
    return (max(lx) + math.log(ss));

def Gibbs_MDGDP(n,R,p1,p2,p3,D,tau,sigma_real,M,burn):
    # --------------------------------定义全局参数-----------------------------------
    
    g = 1;#和delta的超参数有关
    
    hua_A = list();
    grid_n = 10;#网格点的个数
    grid_length = (R**(-0.1)-R**(-D))/grid_n;#网格点的长度
    #grid_length = R**(-1)/grid_n;#网格点的长度
    for i in range(grid_n): hua_A.append((i+1)*grid_length );
    
    MM = 20;#网格点抽样的次数
    
    alpha_0 = 1/R;#默认的迪利克雷浓度
    
    length_nu = 5;
    
    a_nu_grid = [2.1 + i*(D+1-2.1)/(length_nu-1) for i in range(length_nu)];
    
    ceil = math.ceil(10 * R**(1/(2*D))/2)/10
    
    b_nu_grid = [0.5 + i*(ceil-0.5)/(length_nu-1) for i in range(length_nu)];
    
    nu_grid = np.zeros((length_nu**2,2));
    
    for i in range(length_nu):
        
        for j in range(length_nu):
            nu_grid[i*length_nu + j,0] = a_nu_grid[j];
            nu_grid[i*length_nu + j,1] = b_nu_grid[i];
            
    
    #------------------------------输入值计算的结果---------------------------------
    k1 = (1 - 2*tau)/(tau*(1-tau));
    k2 = 2/(tau*(1-tau));
    pd = [p3,p1, p2];
    
    # ---------------------------------------选择先验参数----------------------------
    a_nu = 3*np.ones(R);
    b_nu = 3**(1/(2*D))*np.ones(R);  # 低信息先验
    a_sigma = 1;
    b_sigma = 0.01;
    m = 0;
    v_2 = 1000;
    
    # -------------------------------------生成模拟实验数据--------------------------.
    x = torch.randn([p3, p1, p2])
    factors = parafac(tl.tensor(x), rank=R)
    x_factors = factors.factors
    
    # 循环生成协变量张量
    X = []
    for i in range(n):
        x = torch.randn([p3, p1, p2])
        # x = torch.normal(mean=[1,1,1],std=[3,3,3],[p3,p1,p2]);
        X.append(x)
    
    W = torch.full([p3, p1, p2], 0.0)  # 初始化系数张量 
    # print(W.numpy())
    # 回归系数的真实值
    real = 1;
    
    # 另第一层矩阵的真实值围成三角形
    for i3 in range(p3):
        W[i3, 0:p2, 0:1] = real;
    
    # -----------------------------随机参数观测值yi----------------------------------
    
    y = list()
    q_tau = norm.ppf(q=tau, loc=0, scale=1);
    varepsilon = norm.rvs(loc=-q_tau, scale=sigma_real, size=n)
    for i in range(n):
        inner = torch.sum(torch.mul(W, X[i]))
        y.append(varepsilon[i] + float(inner) )
    #np.percentile(varepsilon, tau)
    # 不需要对每一个协变量进行cp分解
    

    # -----------------------------------初始值-------------------------------------
    # 初始化cp分解后的回归系数张量的期望向量
    iteration_a = list()  # 创建空列表去装
    
    iteration_W = list();
    iteration_nu = list();
    
    for d in range(D):
        mat_1 = 0.1*np.ones( (pd[d],R) );
        iteration_a.append( mat_1 )
        
        ai = list();
        for r in range( R ):
            ai.append( list(1000*np.ones(pd[d])) );
        iteration_W.append( ai )
        iteration_nu.append( list( np.ones((R)) ) );
    
    z = list(np.ones(n));
    sigma = 1;
    intercept = 0;
    
    phi = list(np.ones(R));#收缩CP分解第r个向量的局部参数
    delta = 1; #全局收缩参数
    delta_r = list(np.ones(R));
    
    alpha = 1;
    
    # -----------------------------------抽样序列----------------------------------
    # 在python中直接用=赋值的话一个列表修改了，另一个列表也会跟着更改，为了去除这种关联性
    # 所以采用深度复制函数copy.deepcopy
    sample_a = list(map(lambda x: 0, range(M)));
    sample_a[0] = (iteration_a);
    sample_z = list(map(lambda x: 0, range(M)));
    sample_z.append(z);
    sample_sigma = list(map(lambda x: 0, range(M)));
    sample_sigma.append(sigma);
    sample_intercept = list(map(lambda x: 0, range(M)));
    sample_intercept.append(intercept);
    
    # ----------------------------------Gibbs迭代循环----------------------------------
    #print("数据准备中……\n")
    
    X_d_i = list();
    DD = [2,0,1];
    for d in DD:
        Xd = list();
        for i in range(n):
            x_d_i = X_d(X[i],d);
            Xd.append( x_d_i )
        X_d_i.append(Xd)
    
    #print("数据准备完成。\n")
    
    import time
    start = time.time()
    
    T = 1
    while T < M:
        #print("当前正在进行第{}次迭代。\n".format(T))
        
        #------------------------网格点确定迪利克雷浓度alpha的取值-------------------
    
        alpha = griddy(hua_A,MM,iteration_a,iteration_W,R,D,g,pd)[0];
        
        #alpha = 1/R;
        
        #-------------------------------delta的超参数的设置-------------------------
        
        delta_a = R * alpha;
        
        delta_b = alpha * ( R/g )**(1/D);
        
        # ------------------------------更新回归系数--------------------------------
        for r in range(R):
            
            b_delta_r = 0;
            
            ab_nu_r = griddy_nu(r,phi,delta,R,D,length_nu,nu_grid,iteration_a,pd);
            
            a_nu[r] = ab_nu_r[0];b_nu[r] = ab_nu_r[1];
            
            for d in range(D):
                
                if d == 0:
                    V_d = linalg.khatri_rao(np.array(iteration_a[2]), np.array(iteration_a[1]));
                
                if d == 1:
                    V_d = linalg.khatri_rao(np.array(iteration_a[0]), np.array(iteration_a[2]));
                    
                if d == 2:
                    V_d = linalg.khatri_rao(np.array(iteration_a[0]), np.array(iteration_a[1]));
    
                index_r = list(range(R));
                index_r.remove(r);  # 去除第d个索引
    
                Sigma = np.zeros((pd[d],pd[d]));
                mu = np.matrix(np.zeros((pd[d]))).T;
                for i in range(n):
                    V_d_i = X_d_i[d][i] @ V_d;
                    v_d_i_r = V_d_i[:,r];
    
                    zeta_idr = 0;
                    for ds in index_r:
                        zeta_idr = zeta_idr + float(iteration_a[d][:,ds].T @ V_d_i[:,ds]);
                        
                    mu = mu + v_d_i_r * (y[i] - intercept - zeta_idr - k1*z[i] )/z[i];
                    
                    Sigma = Sigma + v_d_i_r @ v_d_i_r.T/z[i];
                    
                    
                Sigma_dr_inv = Sigma/( k2*sigma ) + np.linalg.inv(np.diag( iteration_W[d][r] ))/( delta * phi[r] );
                
                Sigma_dr = np.linalg.inv(Sigma_dr_inv);
                
                mu_dr = np.array(Sigma_dr @  (mu/(k2*sigma)) )[:, 0];
                
                iteration_a[d][:,r] = np.array(np.random.multivariate_normal(mean=mu_dr,cov = Sigma_dr,size=1 )[0]);
                
                #------------------------更新协方差矩阵W和nu------------------------
                nu_b = sum(abs(iteration_a[d][:,r]))/(phi[r]*delta)**0.5;
                iteration_nu[d][r] = float(np.random.gamma(shape = a_nu[r] + pd[d], scale = 1/(b_nu[r] + nu_b ),  size = 1));
                w_a = iteration_nu[d][r]**2;
                for i_d in range(pd[d]):
                    
                    w_b = iteration_a[d][i_d,r]**2/(phi[r] * delta);
                    iteration_W[d][r][i_d] = float(geninvgauss.rvs(p = 0.5,b = (w_a * w_b)**0.5, scale = (w_b/w_a)**0.5,  size = 1));
                    
                b_delta_r = b_delta_r + iteration_a[d][:,r].T @ np.linalg.inv(np.diag( iteration_W[d][r] )) @ iteration_a[d][:,r];
                
            a_delta_r = 2*delta_b;
            
            delta_r[r] = float(geninvgauss.rvs(p = alpha - 0.5*sum(pd),b = (a_delta_r * b_delta_r)**0.5, scale = (b_delta_r/a_delta_r)**0.5,  size = 1));
                
        sample_a[T] = copy.deepcopy(iteration_a);
        
        sum_delta_r = sum( delta_r );
        
        for r in range(R):
            phi[r] = delta_r[r]/sum_delta_r;
        
        a_delta = 2*delta_b;
        b_delta = 0;
        for r in range(R):
            for d in range(D):
                b_delta = b_delta + iteration_a[d][:,r].T @ np.linalg.inv(np.diag( iteration_W[d][r] )) @ iteration_a[d][:,r]/phi[r];
        
        delta = float(geninvgauss.rvs(p = delta_a - 0.5*R*sum(pd),b = (a_delta * b_delta)**0.5, scale = (b_delta/a_delta)**0.5,  size = 1));
        
        #delta = 100;
        
        #-----------------------------------------更新截距-------------------------
        iteration_B = np.einsum(
            'i,j,k->ijk', np.array(iteration_a[0])[:, 0], np.array(iteration_a[1])[:, 0], np.array(iteration_a[2])[:, 0]);
        for r in range(1, R):
            iteration_B = iteration_B + np.einsum('i,j,k->ijk', np.array(
                iteration_a[0])[:, r], np.array(iteration_a[1])[:, r], np.array(iteration_a[2])[:, r])
    
        iteration_B = torch.tensor(iteration_B)

        # -------------------------------更新zi------------------------------------
        a_z = (k1**2 + 2*k2)/(k2*sigma);
        for i in range(n):
            factor_inner = float(torch.sum(torch.mul(iteration_B, X[i])));
            b_z =  (y[i] - intercept - factor_inner)**2/(k2*sigma) ;
            z_mu = float(geninvgauss.rvs( p = 0.5,b =  (a_z*b_z)**0.5 , scale = (b_z/a_z)**0.5,  size = 1));
            z[i] = z_mu;
        sample_z[T] = copy.deepcopy(z)
        
        # -------------------------------更新sigma----------------------------------
        sss = 0
        for i in range(n):
            factor_inner = float(torch.sum(torch.mul(iteration_B, X[i])));
            sss = sss + (y[i] - intercept - factor_inner - k1*z[i])**2/z[i];
        c_sigma = sss/(2*k2) + sum(z) + b_sigma;
        sigma = float(invgamma.rvs(1.5*n + a_sigma, scale = c_sigma, size = 1 ));
        sample_sigma[T] = copy.deepcopy(sigma);
    
        T = T + 1
    
    end = time.time()
    print("程序的运行时间为{}".format(end-start));
    
    #------------------------------------绘制热力图---------------------------------
    import seaborn as sns #用于画热力图的库
    sample_aa = copy.deepcopy(sample_a);
    MM = burn;
    s_a = sample_a[MM-1];
    
    for d in range(D):
        s_a[d] = np.array(0* np.array(sample_aa[0][d]));
    
    for m in range(MM,M):
        for d in range(D):
            s_a[d] = np.array(s_a[d]) + np.array(sample_aa[m][d]);
    
    mean_a = copy.deepcopy(s_a);
    
    for d in range(D):
        mean_a[d] = s_a[d]/(M-MM);
        
    s_a = sample_aa[MM-1];
    for d in range(D):
        s_a[d] = np.array(0* np.array(sample_aa[0][d]));
        
    for m in range(MM,M):
        for d in range(D):
            for i_d in range(pd[d]):
                for r in range(R):
                    s_a[d][i_d][r] = float(s_a[d][i_d][r] ) + (float(sample_aa[m][d][i_d][r]) - float(mean_a[d][i_d][r]))**2 ;

    var_a = copy.deepcopy(s_a);
    for d in range(D):
        var_a[d] = var_a[d]/(M - MM - 1);
        
    std_a = copy.deepcopy(var_a);
    for d in range(D):
        std_a[d] = var_a[d]**0.5;    
    
    mean_aa = copy.deepcopy( mean_a );
    
    for d in range(D):
        for i_d in range(pd[d]):
            for r in range(R):
                if  0 < mean_a[d][i_d][r] + 1.96*std_a[d][i_d][r] and mean_a[d][i_d][r] - 1.96*std_a[d][i_d][r] <0:
                    mean_aa[d][i_d][r] = 0;
        
    B_estimate = np.einsum(
        'i,j,k->ijk', np.array(mean_aa[0])[:, 0], np.array(mean_aa[1])[:, 0], np.array(mean_aa[2])[:, 0]);
    for r in range(1, R):
        B_estimate = B_estimate + np.einsum('i,j,k->ijk', np.array(
            mean_aa[0])[:, r], np.array(mean_aa[1])[:, r], np.array(mean_aa[2])[:, r])
    
    # torch.sum(torch.mul(np.array(E_a[a,b,c]),X[i]))
    B_estimate = torch.tensor(B_estimate)
    
    B_estimate1 = 0;
    for m in range(MM, M):
        iteration_a = sample_a[m]
        iteration_B = np.einsum(
            'i,j,k->ijk', np.array(iteration_a[0])[:, 0], np.array(iteration_a[1])[:, 0], np.array(iteration_a[2])[:, 0])
        for r in range(1, R):
            iteration_B = iteration_B + np.einsum('i,j,k->ijk', np.array(
                iteration_a[0])[:, r], np.array(iteration_a[1])[:, r], np.array(iteration_a[2])[:, r])
        B_estimate1 = B_estimate1 + iteration_B;
    #iteration_B = torch.tensor(iteration_B)
    B_estimate1 = B_estimate1/(M - MM -1);
    B_estimate1 = torch.tensor(B_estimate1)
    
    EE = float(torch.norm(B_estimate1-W, p = 2)); 
    
    TP = np.sum(np.logical_and(tf.not_equal(W, 0),tf.not_equal(B_estimate, 0))) #True Positive(TP)：预测为正，判断正确；
    FP = np.sum(np.logical_and(tf.equal(W, 0),tf.not_equal(B_estimate, 0))) #False Positive(FP)：预测为正，判断错误；
    TN = np.sum(np.logical_and(tf.equal(W, 0),tf.equal(B_estimate, 0))) #True Negative(TN)：预测为负，判断正确；
    FN = np.sum(np.logical_and(tf.not_equal(W, 0),tf.equal(B_estimate, 0))) #False Negative(FN)：预测为负，判断错误。
    
    return EE,TP,FP,TN,FN

#------------------------------------重复实验-----------------------------------
num = 50;#重复模拟实验的次数
EE_100 = list();TP_100 = list();FP_100 = list();

TN_100 = list();FN_100 = list();

for nm in range(num):
    print("正在运行第{}次重复实验。\n".format(nm+1));
    f = Gibbs_MDGDP(n=800,R=3,p1=50,p2=50,p3=3,D=3,tau=0.25,sigma_real=0.5,M=1000,burn=500)
    EE_100.append(f[0]);TP_100.append(f[1]);FP_100.append(f[2]);
    TN_100.append(f[3]);FN_100.append(f[4]);




