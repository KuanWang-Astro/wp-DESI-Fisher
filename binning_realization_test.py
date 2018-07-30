import numpy as np
import matplotlib.pyplot as plt

##ngal(1)+wp{6+9+14+19+24+29}+ggl{7+10+15+20+25+30}+vpf{7+10+15+20+25+30}+Pcic{10+15+25+40+55+70}

class process_test(object):
    def __init__(self, zpfile, pertfile):
        self.zp = zpfile
        self.pert = pertfile
        self.assign()
        self.calc_cov()
        
    def assign(self):
        self.p0 = self.zp['param'][0]
        self.ppert = self.pert['param']
        self.npert = np.sum(self.pert['i']==0)
        #1: 0:7   7:14 14:21 21:31
        #2: 0:10 10:20 20:30 30:45
        #3: 0:15 15:30 30:45 45:70
        #4: 0:20 20:40 40:60 60:100
        #5: 0:25 25:50 50:75 75:130
        #6: 0:30 30:60 60:90 90:160
        self.wp7_0 = self.zp['func_all_1'][:,0:7]
        self.wp10_0 = self.zp['func_all_2'][:,0:10]
        self.wp15_0 = self.zp['func_all_3'][:,0:15]
        self.wp20_0 = self.zp['func_all_4'][:,0:20]
        self.wp25_0 = self.zp['func_all_5'][:,0:25]
        self.wp30_0 = self.zp['func_all_6'][:,0:30]
        ###
        self.wp7_p = self.pert['func_all_1'][:,0:7]
        self.wp10_p = self.pert['func_all_2'][:,0:10]
        self.wp15_p = self.pert['func_all_3'][:,0:15]
        self.wp20_p = self.pert['func_all_4'][:,0:20]
        self.wp25_p = self.pert['func_all_5'][:,0:25]
        self.wp30_p = self.pert['func_all_6'][:,0:30]
        ###
        self.ds7_0 = self.zp['func_all_1'][:,7:14]
        self.ds10_0 = self.zp['func_all_2'][:,10:20]
        self.ds15_0 = self.zp['func_all_3'][:,15:30]
        self.ds20_0 = self.zp['func_all_4'][:,20:40]
        self.ds25_0 = self.zp['func_all_5'][:,25:50]
        self.ds30_0 = self.zp['func_all_6'][:,30:60]
        ###
        self.ds7_p = self.pert['func_all_1'][:,7:14]
        self.ds10_p = self.pert['func_all_2'][:,10:20]
        self.ds15_p = self.pert['func_all_3'][:,15:30]
        self.ds20_p = self.pert['func_all_4'][:,20:40]
        self.ds25_p = self.pert['func_all_5'][:,25:50]
        self.ds30_p = self.pert['func_all_6'][:,30:60]
        ###
        self.vpf7_0 = self.zp['func_all_1'][:,14:21]
        self.vpf10_0 = self.zp['func_all_2'][:,20:30]
        self.vpf15_0 = self.zp['func_all_3'][:,30:45]
        self.vpf20_0 = self.zp['func_all_4'][:,40:60]
        self.vpf25_0 = self.zp['func_all_5'][:,50:75]
        self.vpf30_0 = self.zp['func_all_6'][:,60:90]
        ###
        self.vpf7_p = self.pert['func_all_1'][:,14:21]
        self.vpf10_p = self.pert['func_all_2'][:,20:30]
        self.vpf15_p = self.pert['func_all_3'][:,30:45]
        self.vpf20_p = self.pert['func_all_4'][:,40:60]
        self.vpf25_p = self.pert['func_all_5'][:,50:75]
        self.vpf30_p = self.pert['func_all_6'][:,60:90]
        ###
        self.cic10_0 = self.zp['func_all_1'][:,21:]
        self.cic15_0 = self.zp['func_all_2'][:,30:]
        self.cic25_0 = self.zp['func_all_3'][:,45:]
        self.cic40_0 = self.zp['func_all_4'][:,60:]
        self.cic55_0 = self.zp['func_all_5'][:,75:]
        self.cic70_0 = self.zp['func_all_6'][:,90:]
        ###
        self.cic10_p = self.pert['func_all_1'][:,21:]
        self.cic15_p = self.pert['func_all_2'][:,30:]
        self.cic25_p = self.pert['func_all_3'][:,45:]
        self.cic40_p = self.pert['func_all_4'][:,60:]
        self.cic55_p = self.pert['func_all_5'][:,75:]
        self.cic70_p = self.pert['func_all_6'][:,90:]
        ###
        self.wp7_cov = self.zp['func_all_cov_1'][:,0:7,0:7]
        self.wp10_cov = self.zp['func_all_cov_2'][:,0:10,0:10]
        self.wp15_cov = self.zp['func_all_cov_3'][:,0:15,0:15]
        self.wp20_cov = self.zp['func_all_cov_4'][:,0:20,0:20]
        self.wp25_cov = self.zp['func_all_cov_5'][:,0:25,0:25]
        self.wp30_cov = self.zp['func_all_cov_6'][:,0:30,0:30]
        ###
        self.ds7_cov = self.zp['func_all_cov_1'][:,7:14,7:14]
        self.ds10_cov = self.zp['func_all_cov_2'][:,10:20,10:20]
        self.ds15_cov = self.zp['func_all_cov_3'][:,15:30,15:30]
        self.ds20_cov = self.zp['func_all_cov_4'][:,20:40,20:40]
        self.ds25_cov = self.zp['func_all_cov_5'][:,25:50,25:50]
        self.ds30_cov = self.zp['func_all_cov_6'][:,30:60,30:60]
        ###
        self.vpf7_cov = self.zp['func_all_cov_1'][:,14:21,14:21]
        self.vpf10_cov = self.zp['func_all_cov_2'][:,20:30,20:30]
        self.vpf15_cov = self.zp['func_all_cov_3'][:,30:45,30:45]
        self.vpf20_cov = self.zp['func_all_cov_4'][:,40:60,40:60]
        self.vpf25_cov = self.zp['func_all_cov_5'][:,50:75,50:75]
        self.vpf30_cov = self.zp['func_all_cov_6'][:,60:90,60:90]
        ###
        self.cic10_cov = self.zp['func_all_cov_1'][:,21:,21:]
        self.cic15_cov = self.zp['func_all_cov_2'][:,30:,30:]
        self.cic25_cov = self.zp['func_all_cov_3'][:,45:,45:]
        self.cic40_cov = self.zp['func_all_cov_4'][:,60:,60:]
        self.cic55_cov = self.zp['func_all_cov_5'][:,75:,75:]
        self.cic70_cov = self.zp['func_all_cov_6'][:,90:,90:]
        
    def calc_cov(self):
        self.wp7_cr = np.cov(self.wp7_0.T)
        self.wp7_cj = np.mean(self.wp7_cov,axis=0)
        self.wp7_ct = self.wp7_cr+self.wp7_cj
        self.wp10_cr = np.cov(self.wp10_0.T)
        self.wp10_cj = np.mean(self.wp10_cov,axis=0)
        self.wp10_ct = self.wp10_cr+self.wp10_cj
        self.wp15_cr = np.cov(self.wp15_0.T)
        self.wp15_cj = np.mean(self.wp15_cov,axis=0)
        self.wp15_ct = self.wp15_cr+self.wp15_cj
        self.wp20_cr = np.cov(self.wp20_0.T)
        self.wp20_cj = np.mean(self.wp20_cov,axis=0)
        self.wp20_ct = self.wp20_cr+self.wp20_cj
        self.wp25_cr = np.cov(self.wp25_0.T)
        self.wp25_cj = np.mean(self.wp25_cov,axis=0)
        self.wp25_ct = self.wp25_cr+self.wp25_cj
        self.wp30_cr = np.cov(self.wp30_0.T)
        self.wp30_cj = np.mean(self.wp30_cov,axis=0)
        self.wp30_ct = self.wp30_cr+self.wp30_cj
        ###
        self.ds7_cr = np.cov(self.ds7_0.T)
        self.ds7_cj = np.mean(self.ds7_cov,axis=0)
        self.ds7_ct = self.ds7_cr+self.ds7_cj
        self.ds10_cr = np.cov(self.ds10_0.T)
        self.ds10_cj = np.mean(self.ds10_cov,axis=0)
        self.ds10_ct = self.ds10_cr+self.ds10_cj
        self.ds15_cr = np.cov(self.ds15_0.T)
        self.ds15_cj = np.mean(self.ds15_cov,axis=0)
        self.ds15_ct = self.ds15_cr+self.ds15_cj
        self.ds20_cr = np.cov(self.ds20_0.T)
        self.ds20_cj = np.mean(self.ds20_cov,axis=0)
        self.ds20_ct = self.ds20_cr+self.ds20_cj
        self.ds25_cr = np.cov(self.ds25_0.T)
        self.ds25_cj = np.mean(self.ds25_cov,axis=0)
        self.ds25_ct = self.ds25_cr+self.ds25_cj
        self.ds30_cr = np.cov(self.ds30_0.T)
        self.ds30_cj = np.mean(self.ds30_cov,axis=0)
        self.ds30_ct = self.ds30_cr+self.ds30_cj
        ###
        self.vpf7_cr = np.cov(self.vpf7_0.T)
        self.vpf7_cj = np.mean(self.vpf7_cov,axis=0)
        self.vpf7_ct = self.vpf7_cr+self.vpf7_cj
        self.vpf10_cr = np.cov(self.vpf10_0.T)
        self.vpf10_cj = np.mean(self.vpf10_cov,axis=0)
        self.vpf10_ct = self.vpf10_cr+self.vpf10_cj
        self.vpf15_cr = np.cov(self.vpf15_0.T)
        self.vpf15_cj = np.mean(self.vpf15_cov,axis=0)
        self.vpf15_ct = self.vpf15_cr+self.vpf15_cj
        self.vpf20_cr = np.cov(self.vpf20_0.T)
        self.vpf20_cj = np.mean(self.vpf20_cov,axis=0)
        self.vpf20_ct = self.vpf20_cr+self.vpf20_cj
        self.vpf25_cr = np.cov(self.vpf25_0.T)
        self.vpf25_cj = np.mean(self.vpf25_cov,axis=0)
        self.vpf25_ct = self.vpf25_cr+self.vpf25_cj
        self.vpf30_cr = np.cov(self.vpf30_0.T)
        self.vpf30_cj = np.mean(self.vpf30_cov,axis=0)
        self.vpf30_ct = self.vpf30_cr+self.vpf30_cj
        ###
        self.cic10_cr = np.cov(self.cic10_0.T)
        self.cic10_cj = np.mean(self.cic10_cov,axis=0)
        self.cic10_ct = self.cic10_cr+self.cic10_cj
        self.cic15_cr = np.cov(self.cic15_0.T)
        self.cic15_cj = np.mean(self.cic15_cov,axis=0)
        self.cic15_ct = self.cic15_cr+self.cic15_cj
        self.cic25_cr = np.cov(self.cic25_0.T)
        self.cic25_cj = np.mean(self.cic25_cov,axis=0)
        self.cic25_ct = self.cic25_cr+self.cic25_cj
        self.cic40_cr = np.cov(self.cic40_0.T)
        self.cic40_cj = np.mean(self.cic40_cov,axis=0)
        self.cic40_ct = self.cic40_cr+self.cic40_cj
        self.cic55_cr = np.cov(self.cic55_0.T)
        self.cic55_cj = np.mean(self.cic55_cov,axis=0)
        self.cic55_ct = self.cic55_cr+self.cic55_cj
        self.cic70_cr = np.cov(self.cic70_0.T)
        self.cic70_cj = np.mean(self.cic70_cov,axis=0)
        self.cic70_ct = self.cic70_cr+self.cic70_cj
        
    def plot_cov(self, cov):
        plt.imshow((cov/np.sqrt(cov.diagonal())).T/np.sqrt(cov.diagonal()),\
                   vmin=-1, vmax=1, cmap='seismic')
        plt.colorbar()
        
    def calc_dfdp(self, func, func0):
        dfdp = np.zeros((7,func.shape[1]))
        for i in range(7):
            for j in range(func.shape[1]):
                dfdp[i,j] = np.linalg.lstsq((self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i])[:,np.newaxis],\
                           func[self.npert*i:self.npert*(i+1),j]-np.mean(func0[:,j],axis=0))[0]
        return dfdp
    
    def calc_fisher(self, dfdp, covtot):
        fmatrix = np.zeros((7,7))
        for i in range(7):
            for j in range(7):
                fmatrix[i,j] = np.dot(dfdp[i],np.dot(np.linalg.inv(covtot),dfdp[j]))
        return fmatrix
    
    def calc_1sigma(self, func, func0, covtot):
        return np.sqrt(np.linalg.inv(self.calc_fisher(self.calc_dfdp(func,func0),covtot)).diagonal())
    
    def plot_dfdp(self, func, func0, i, j):
        plt.plot(self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i],\
                         func[self.npert*i:self.npert*(i+1),j]-np.mean(func0[:,j],axis=0),'.')
        plt.plot(func0[:,j]*0.,func0[:,j]-np.mean(func0[:,j],axis=0),'r^')
        plt.axhline(0,c='k',linestyle='--')
        plt.axvline(0,c='k',linestyle='--')
        plt.plot(np.array((min(self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i]),\
                           max(self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i]))),\
                 np.array((min(self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i]),\
                           max(self.ppert[self.npert*i:self.npert*(i+1),i]-self.p0[i])))*self.calc_dfdp(func,func0,self.ppert)[i,j])
        
    
        
    def calc_cov_fewer_real(self, func, covjk):
        cr = np.cov(func.T)
        cj = np.mean(covjk,axis=0)
        ct = cr+cj
        return ct
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        