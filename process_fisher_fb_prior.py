import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class process_fisher(object):
    def __init__(self, zpfunc, zpcov, zpparam, psfunc, psparam):
        self.zpfunc = zpfunc
        self.len_obs = self.zpfunc[0].size
        self.zpfunccov = zpcov
        self.zpparam = zpparam
        self.psfunc = psfunc
        self.psparam = psparam
        self.Nps = self.psparam.size/49
        
    def calc_cov(self):
        self.cov_real = np.cov(self.zpfunc.T)
        self.cov_jk = np.mean(self.zpfunccov,axis=0)
        self.cov_tot = self.cov_real+self.cov_jk
    
    def plot_cov(self):
        tick_cic = (self.len_obs+90)/2
        plt.figure()
        plt.title('Correlation matrix across realizations',y=1.08)
        plt.imshow((self.cov_real/np.sqrt(self.cov_real.diagonal())).T/np.sqrt(self.cov_real.diagonal()),\
                   vmin=-1, vmax=1, cmap='bwr')
        plt.xticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.yticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.tick_params(length=0,labeltop='on',labelbottom='off')
        plt.colorbar()
        
        plt.figure()
        plt.title('Correlation matrix from jackknife',y=1.08)
        plt.imshow((self.cov_jk/np.sqrt(self.cov_jk.diagonal())).T/np.sqrt(self.cov_jk.diagonal()),\
                   vmin=-1, vmax=1, cmap='bwr')
        plt.xticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.yticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.tick_params(length=0,labeltop='on',labelbottom='off')
        plt.colorbar()
        
        plt.figure()
        plt.title('Total correlation matrix',y=1.08)
        plt.imshow((self.cov_tot/np.sqrt(self.cov_tot.diagonal())).T/np.sqrt(self.cov_tot.diagonal()),\
                   vmin=-1, vmax=1, cmap='bwr')
        plt.xticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.yticks((0,15,45,75,tick_cic),(r'$n_g$',r'$w_p(r_p)$',r'$\Delta\Sigma(r_p)$',r'$vpf$',r'$P(N_{cic})$'))
        plt.tick_params(length=0,labeltop='on',labelbottom='off')
        plt.colorbar()
    
    def calc_dfdp(self):
        self.dfdp = np.zeros((7,self.len_obs))
        for i in range(7):
            for j in range(self.len_obs):
                self.dfdp[i,j] = np.linalg.lstsq((self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i])[:,np.newaxis],\
                           self.psfunc[self.Nps*i:self.Nps*(i+1),j]-np.mean(self.zpfunc[:,j],axis=0))[0]
        

    def plot_dfdp(self):
        for i in range(7):
            for j in range(self.len_obs):
                plt.figure()
                plt.plot(self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i],\
                         self.psfunc[self.Nps*i:self.Nps*(i+1),j]-np.mean(self.zpfunc[:,j],axis=0),'.',markersize=0.2)
                plt.plot(self.zpfunc[:,j]*0.,self.zpfunc[:,j]-np.mean(self.zpfunc[:,j],axis=0),'r^',markersize=1)
                plt.xlabel('dp'+str(i))
                plt.ylabel('df'+str(j))
                plt.axhline(0,c='k',linestyle='--')
                plt.plot(np.array((min(self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i]),\
                                   max(self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i]))),\
                         np.array((min(self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i]),\
                                   max(self.psparam[self.Nps*i:self.Nps*(i+1),i]-self.zpparam[i])))*self.dfdp[i,j])
                plt.savefig(str(self.outfolder+'/p{}f{}').format(i,j))
                plt.close()
                
    def calc_fisher(self, dfdp, covtot):
        fmatrix = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                fmatrix[i,j] = np.dot(dfdp[i],np.dot(np.linalg.inv(covtot),dfdp[j]))
        return fmatrix
                
    def plot_norm_inv_fisher(self, fmatrix, title):
        plt.figure()
        plt.imshow((np.linalg.inv(fmatrix)/np.sqrt(np.linalg.inv(fmatrix).diagonal())).T\
           /np.sqrt(np.linalg.inv(fmatrix).diagonal()),vmin=-1,vmax=1,cmap='bwr')
        plt.title(title)
        plt.colorbar()
        
    def calc_1sigma(self, fmatrix):
        return np.sqrt(np.linalg.inv(fmatrix).diagonal())
    
    def split_sects(self):
        self.dfdp_wp = self.dfdp[:,:30]
        self.dfdp_ds = self.dfdp[:,30:60]
        self.dfdp_vpf = self.dfdp[:,60:90]
        self.dfdp_cic = self.dfdp[:,90:self.len_obs]
        
        self.cov_ww = self.cov_tot[:30,:30]
        self.cov_wd = self.cov_tot[:30,30:60]
        self.cov_wv = self.cov_tot[:30,60:90]
        self.cov_wc = self.cov_tot[:30,90:self.len_obs]
        self.cov_dw = self.cov_tot[30:60,:30]
        self.cov_dd = self.cov_tot[30:60,30:60]
        self.cov_dv = self.cov_tot[30:60,60:90]
        self.cov_dc = self.cov_tot[30:60,90:self.len_obs]
        self.cov_vw = self.cov_tot[60:90,:30]
        self.cov_vd = self.cov_tot[60:90,30:60]
        self.cov_vv = self.cov_tot[60:90,60:90]
        self.cov_vc = self.cov_tot[60:90,90:self.len_obs]
        self.cov_cw = self.cov_tot[90:self.len_obs,:30]
        self.cov_cd = self.cov_tot[90:self.len_obs,30:60]
        self.cov_cv = self.cov_tot[90:self.len_obs,60:90]
        self.cov_cc = self.cov_tot[90:self.len_obs,90:self.len_obs]

    def sort_comb(self):
        self.constraint = np.zeros((8,7))
        self.constraint[0] = self.one_sigma_w
        self.constraint[1] = self.one_sigma_wd
        self.constraint[2] = self.one_sigma_wv
        self.constraint[3] = self.one_sigma_wc
        self.constraint[4] = self.one_sigma_wdv
        self.constraint[5] = self.one_sigma_wdc
        self.constraint[6] = self.one_sigma_wvc
        self.constraint[7] = self.one_sigma_wdvc
        for i in range(7):
            print((self.constraint[:,i]).argsort()) 

            
    def plot_ellipse(self,fisher,i,j):
        plt.figure()
        cov = np.linalg.inv(fisher)
        cov2d = np.array(((cov[i,i],cov[i,j]),(cov[j,i],cov[j,j])))
        center = (self.zpparam[i],self.zpparam[j])
        eigval, eigvec = np.linalg.eig(cov2d)
        e = Ellipse(center, 2*eigval[0], 2*eigval[1], np.arctan(eigvec[0][1]/eigvec[0][0])*180/np.pi)
        a = plt.subplot(111)
        a.axis('equal')
        e.set_edgecolor('k')
        e.set_facecolor('w')
        a.add_artist(e)
        plt.xlabel(self.paramlist[i])
        plt.ylabel(self.paramlist[j])
        
        
    def process(self, plot_dfdp=False, outfolder=None, len_to_use=160,):
        self.paramlist = ['alpha', 'logM1', 'sigma_logM', 'logM0', 'logMmin', 'Acen', 'Asat']
        self.calc_cov()
        if len_to_use!=160:
            self.len_obs = len_to_use
            self.cov_real = self.cov_real[:self.len_obs,:self.len_obs]
            self.cov_jk = self.cov_jk[:self.len_obs,:self.len_obs]
            self.cov_tot = self.cov_tot[:self.len_obs,:self.len_obs]
        self.plot_cov()
        self.calc_dfdp()
        if plot_dfdp and outfolder!=None:
            self.outfolder = outfolder
            self.plot_dfdp()
        
        self.split_sects()
        
        self.fishers = []
        
        ###
        fisher = self.calc_fisher(self.dfdp_wp,self.cov_ww)
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p only$')
        self.one_sigma_w = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_ds),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wd),axis=1),\
                                                  np.concatenate((self.cov_dw,self.cov_dd),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+\Delta\Sigma$')
        self.one_sigma_wd = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_vpf),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wv),axis=1),\
                                                  np.concatenate((self.cov_vw,self.cov_vv),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+vpf$')
        self.one_sigma_wv = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_cic),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wc),axis=1),\
                                                  np.concatenate((self.cov_cw,self.cov_cc),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+P(N_{cic})$')
        self.one_sigma_wc = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_ds,self.dfdp_vpf),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wd,self.cov_wv),axis=1),\
                                                  np.concatenate((self.cov_dw,self.cov_dd,self.cov_dv),axis=1),\
                                                  np.concatenate((self.cov_vw,self.cov_vd,self.cov_vv),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+\Delta\Sigma+vpf$')
        self.one_sigma_wdv = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_ds,self.dfdp_cic),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wd,self.cov_wc),axis=1),\
                                                  np.concatenate((self.cov_dw,self.cov_dd,self.cov_dc),axis=1),\
                                                  np.concatenate((self.cov_cw,self.cov_cd,self.cov_cc),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+\Delta\Sigma+P(N_{cic})$')
        self.one_sigma_wdc = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(np.concatenate((self.dfdp_wp,self.dfdp_vpf,self.dfdp_cic),axis=1),\
                                  np.concatenate((np.concatenate((self.cov_ww,self.cov_wv,self.cov_wc),axis=1),\
                                                  np.concatenate((self.cov_vw,self.cov_vv,self.cov_vc),axis=1),\
                                                  np.concatenate((self.cov_cw,self.cov_cv,self.cov_cc),axis=1))))
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+vpf+P(N_{cic})$')
        self.one_sigma_wvc = self.calc_1sigma(fisher)
        ###
        fisher = self.calc_fisher(self.dfdp,self.cov_tot)
        self.fishers.append(fisher)
        self.plot_norm_inv_fisher(fisher,r'$n_g+w_p+\Delta\Sigma+vpf+P(N_{cic})$')
        self.one_sigma_wdvc = self.calc_1sigma(fisher)
        
        self.sort_comb()
        
        ###
        self.fisher_d = self.calc_fisher(self.dfdp_ds,self.cov_dd)
        self.plot_norm_inv_fisher(self.fisher_d,r'$\Delta\Sigma$')
        self.one_sigma_d = self.calc_1sigma(self.fisher_d)
        ###
        self.fisher_v = self.calc_fisher(self.dfdp_vpf,self.cov_vv)
        self.plot_norm_inv_fisher(self.fisher_v,r'$vpf$')
        self.one_sigma_v = self.calc_1sigma(self.fisher_v)
        ###
        self.fisher_c = self.calc_fisher(self.dfdp_cic,self.cov_cc)
        self.plot_norm_inv_fisher(self.fisher_c,r'$P(N_{cic})$')
        self.one_sigma_c = self.calc_1sigma(self.fisher_c)
        
        
        return self.constraint,self.one_sigma_d,self.one_sigma_v,self.one_sigma_c