# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:07:19 2023

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

from scipy.stats import norm
from scipy.stats import invgamma  #从逆伽马分布中抽样
from scipy.stats import invgauss  #从逆高斯分布中抽样
from scipy.stats import binom  #从两点分布中抽样
from scipy.stats import geninvgauss  #从广义逆高斯分布中抽样

from scipy.stats import beta  #从贝塔分布中抽样

#---------------------------------调用GPU执行代码-------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

# -------------------------------需要用到的函数----------------------------------

def vec(X):  # 将张量拉直为列向量的函数
    vec_X = list()
    for i3 in range(len(X)):
        for r in range(len(X[i3][0,:])):
            vec_X = vec_X + list(np.array(X[i3][:, r]))
    return vec_X


def vec_X_id(X, d, i_d):
    X = list(np.array(X))
    
    if d == 1:
        for i3 in range(len(X)):
            X[i3] = X[i3][i_d, :]
    if d == 2:
        for i3 in range(len(X)):
            X[i3] = X[i3][:, i_d]
        
    if d == 0:
        X = X[i_d]
    X = torch.tensor(X);
    vec_X_id = tf.reshape(X, [-1]);
    return vec_X_id


def vec_X_minus_id(X, d, i_d):  # 将张量拉直为列向量的函数
    Z = np.array(X);
    X = list(np.array(X))
    #vec_X_minus_id = list()
    if d == 1:
        for i3 in range(len(X)):
            X[i3] = np.delete(Z[i3], i_d, axis=0);
            
    if d == 2:
        for i3 in range(len(X)):
            X[i3] = np.delete(Z[i3], i_d, axis=1);
            
    if d == 0:
        X.pop(i_d);
    X = torch.tensor(X);
    vec_X_minus_id = tf.reshape(X, [-1]);
    return vec_X_minus_id

def Gibbs_MDGDP(n,R,p1,p2,p3,D,tau,sigma_real,M,burn):
    
    #------------------------------输入值计算的结果---------------------------------
    k1 = (1 - 2*tau)/(tau*(1-tau));
    k2 = 2/(tau*(1-tau));
    
    # ---------------------------------------选择先验参数----------------------------
    a_tau = 1
    b_tau = 1  # 低信息先验
    a_rho = 1
    b_rho = 1
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
        W[i3, 0:1, 0:p1] = real;
    
    # -----------------------------随机参数观测值yi----------------------------------
    # torch.sum(torch.mul(W,x)) 表示W和x的内积
    #random.seed(5408)
    
    y = list();
    q_tau = norm.ppf(q=tau, loc=0, scale=sigma_real);
    varepsilon = norm.rvs(loc=-q_tau, scale=sigma_real, size=n);
    
    for i in range(n):
        inner = torch.sum(torch.mul(W, X[i]))
        y.append(varepsilon[i] + float(inner) )
    
    
    # -----------------------------------初始值-------------------------------------
    # 初始化cp分解后的回归系数张量的期望向量
    iteration_a = list()  # 创建空列表去装
    iteration_Lambda = list()
    iteration_Lambda_xing = list()
    iteration_Lambda_xing_xing = list();
    iteration_inv_Lambda = list()
    Sigma_a = list()
    iteration_a_xing = list()
    iteration_vartheta = list(np.ones(D));
    r_id = list();
    pd = [p3, p1, p2]
    for d in range(D):
        ai = list()
        bi = list()
        ci = list()
        di = list()
        ei = list()
        fi = list()
        hi = list()
        ki = list();
        qi = list();
        for i in range(pd[d]):
            ai.append(list(np.ones(R)))
            bi.append(np.diag((np.ones(R))))
            ci.append(list(np.ones(R)))
            di.append(list(np.ones(R)))
            ei.append(list(1*np.ones(R)))
            fi.append(list(1*np.ones(R)))
            ki.append(list(np.ones(R)));
            qi.append(list(np.ones(R)));
        iteration_a.append(list(ei))
        iteration_Lambda.append(list(ai))
        iteration_Lambda_xing.append(list(di))
        iteration_Lambda_xing_xing.append(list(qi))
        iteration_inv_Lambda.append(list(ci))
        iteration_a_xing.append(list(fi))
        Sigma_a.append(list(bi))
        r_id.append(list(ki));
    
    z = list(np.ones(n));
    sigma = 1;
    intercept = 0;
    rho = list(0.5*np.ones(D));
    
    # -----------------------------------抽样序列-----------------------------------
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
    
    mse = 0;
    for i in range(n):
        inner = torch.sum(torch.mul(W, X[i]))
        mse = mse + (y[i] - intercept - inner)
    mse = mse/n;
    
    
    # ----------------------------------Gibbs迭代循环----------------------------------
    #print("数据准备中……\n")
    vec_Xi = list()
    vec_Xi_minus = list()
    for d in range(D):
        Xd = list();Xd_minus = list();
        for i_d in range(pd[d]):
            Xi = list();Xi_minus = list();
            for i in range(n):
                vec_Xi_id = vec_X_id(X[i], d, i_d);  # 按列拉直
                vec_Xi_minus_id = vec_X_minus_id(X[i], d, i_d);  # 按列拉直
                vec_Xi_id = np.matrix(vec_Xi_id).T
                Xi.append(vec_Xi_id );
                vec_Xi_minus_id = np.matrix(vec_Xi_minus_id).T
                Xi_minus.append(vec_Xi_minus_id)
                
            Xd.append(Xi);Xd_minus.append(Xi_minus);
        vec_Xi.append(Xd);
        vec_Xi_minus.append(Xd_minus);
        
    #print("数据准备完成。\n")
    
    T = 1
    while T < M:
        #print("当前正在进行第{}次迭代。\n".format(T))
        # ------------------------------更新回归系数--------------------------------
        for d in range(D):
            index = list(range(D))
            index.remove(d)  # 去除第d个索引
            eta = np.array(iteration_a[index[0]]);
            index_1 = index[1:]  # 除去索引中的第一个元素
            for t in index_1:
                eta = linalg.khatri_rao(eta, np.array(iteration_a[t]));
            for i_d in range(pd[d]):
                iteration_zi_etai = np.zeros((R, R));  # 创建R行R列的零矩阵
                mu = 0
                A_minus_id = np.delete(iteration_a[d], [i_d], axis=0);
                iteration_a_minus_d = list(iteration_a);
                iteration_a_minus_d[d] = A_minus_id;
    
                index_2 = list(range(D))
                iteration_A_minus_id = np.array(iteration_a_minus_d[index_2[0]])
                for dt in index_2[1:]:
                    iteration_A_minus_id = linalg.khatri_rao(iteration_A_minus_id, np.array(iteration_a_minus_d[dt]));
                for i in range(n):
                    vec_Xi_id = vec_Xi[d][i_d][i];
                    vec_Xi_minus_id = vec_Xi_minus[d][i_d][i];
                    
                    iteration_zi_etai = iteration_zi_etai + (eta).T @ vec_Xi_id @ vec_Xi_id.T @ (eta)/z[i];
                    
                    varthetai = (( (iteration_A_minus_id) @ np.ones(R)).T @ vec_Xi_minus_id)[0, 0];
                    
                    mu = mu + ( (eta).T @ vec_Xi_id ) * (y[i] - intercept - varthetai - k1*z[i])/z[i];
                    
                Sigma = iteration_zi_etai/(k2*sigma) + np.diag(iteration_inv_Lambda[d][i_d]);
                Sigma_a[d][i_d] = np.linalg.inv(Sigma);
                mu_xing = np.array(Sigma_a[d][i_d] @  (mu/(k2*sigma)) )[:, 0];
                #diag_Sigma_a[d][i_d] = np.diag(Sigma_a[d][i_d]);
                iteration_a_xing[d][i_d] = np.array(np.random.multivariate_normal(mean=mu_xing,cov = Sigma_a[d][i_d],size=1 )[0]);
    
                # ----------------------------Lambda-------------------------------
                Lambda_xing = 1;
                for r in range(R):
                    lambda_a = iteration_vartheta[d];
                    lambda_b = iteration_a_xing[d][i_d][r]**2;
                    iteration_Lambda_xing[d][i_d][r] = float(geninvgauss.rvs(p = 0.5,b = (lambda_a * lambda_b)**0.5, scale = (lambda_b/lambda_a)**0.5,  size = 1));
                    iteration_Lambda_xing_xing[d][i_d][r] = float(np.random.exponential(2/lambda_a,1));
                    #iteration_Lambda_xing_xing[d][i_d][r] = 0;
                    Lambda_xing = Lambda_xing*iteration_Lambda_xing[d][i_d][r];
                iteration_inv_Lambda[d][i_d] = 1/np.array(iteration_Lambda_xing[d][i_d]) ;

                Sigma = iteration_zi_etai/(k2*sigma) + np.diag(iteration_inv_Lambda[d][i_d]);
                Sigma_a[d][i_d] = np.linalg.inv(Sigma);
                mu_xing = np.array(np.linalg.inv(Sigma) @  (mu/(k2*sigma)) )[:, 0];
                zeta_id = math.log(1 - rho[d]) - math.log(rho[d]) + 0.5* math.log(Lambda_xing) - \
                    0.5*math.log( np.linalg.det(Sigma_a[d][i_d]) ) - 0.5*mu_xing @ np.linalg.inv(Sigma_a[d][i_d]) @ mu_xing.T;
                try:
                    r_id[d][i_d] = binom.rvs(1,1/(1+math.exp(float(zeta_id))));
                except:
                    r_id[d][i_d] = float(binom.rvs(1,0));
                iteration_a[d][i_d] = iteration_a_xing[d][i_d] * float(r_id[d][i_d]);
                iteration_Lambda[d][i_d] = float(r_id[d][i_d]) * np.array(iteration_Lambda_xing[d][i_d]) + (1 - float(r_id[d][i_d]))*np.array(iteration_Lambda_xing_xing[d][i_d]);
                
                
        # ------------------------------vartheta和rho--------------------------------
        
        for dd in range(D):
            s_gamma = 0;s_h2 = 0;
            for i_dd in range(pd[dd]):
                s_gamma = s_gamma + r_id[dd][i_dd];
                for rr in range(R):
                    #s_q_zjdr = s_q_zjdr + q_zjd[dd][i_dd][rr];
                    s_h2 = s_h2 + iteration_Lambda[dd][i_dd][rr];
            #s_q_zjdr = s_q_zjdr/3;
            rho[dd] = beta.rvs(a_rho + s_gamma ,b_rho + pd[dd] - s_gamma,size = 1 );
            iteration_vartheta[dd] = float(np.random.gamma(shape = a_tau + R*pd[dd], scale = 1/(b_tau +s_h2/2),  size = 1));
        
        sample_a[T] = copy.deepcopy(iteration_a);
    
        #iteration_a = W_factors;
        #-----------------------------------------更新截距-------------------------
        iteration_B = np.einsum(
            'i,j,k->ijk', np.array(iteration_a[0])[:, 0], np.array(iteration_a[1])[:, 0], np.array(iteration_a[2])[:, 0]);
        for r in range(1, R):
            iteration_B = iteration_B + np.einsum('i,j,k->ijk', np.array(
                iteration_a[0])[:, r], np.array(iteration_a[1])[:, r], np.array(iteration_a[2])[:, r])
    
        # torch.sum(torch.mul(np.array(E_a[a,b,c]),X[i]))
        iteration_B = torch.tensor(iteration_B)
        #iteration_B = W;
        
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
    
    EE = float(torch.norm(B_estimate-W, p = 2)); 
    
    EE1 = float(torch.norm(B_estimate1-W, p = 2)); 
    
    TP = np.sum(np.logical_and(tf.not_equal(W, 0),tf.not_equal(B_estimate, 0))) #True Positive(TP)：预测为正，判断正确；
    FP = np.sum(np.logical_and(tf.equal(W, 0),tf.not_equal(B_estimate, 0))) #False Positive(FP)：预测为正，判断错误；
    TN = np.sum(np.logical_and(tf.equal(W, 0),tf.equal(B_estimate, 0))) #True Negative(TN)：预测为负，判断正确；
    FN = np.sum(np.logical_and(tf.not_equal(W, 0),tf.equal(B_estimate, 0))) #False Negative(FN)：预测为负，判断错误。
    
    return EE,TP,FP,TN,FN,EE1


#------------------------------------重复实验-----------------------------------
num = 50;#重复模拟实验的次数
EE_100 = list();TP_100 = list();FP_100 = list();

TN_100 = list();FN_100 = list();EE1_100 = list();

#f = Gibbs_MDGDP(n=1000,R=2,p1=5,p2=5,p3=5,D=3,tau=0.5,sigma_real=1,M=1000,burn=500)

for nm in range(num):
    print("正在运行第{}次重复实验。\n".format(nm+1));
    f = Gibbs_MDGDP(n=500,R=3,p1=5,p2=5,p3=5,D=3,tau=0.5,sigma_real=1,M=1000,burn=500)
    EE_100.append(f[0]);TP_100.append(f[1]);FP_100.append(f[2]);
    TN_100.append(f[3]);FN_100.append(f[4]);EE1_100.append(f[5]);
    


