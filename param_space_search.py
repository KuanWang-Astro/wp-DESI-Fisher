import argparse

parser = argparse.ArgumentParser(description='#')
parser.add_argument('--Lbox',type=int,required=True,dest='Lbox')
#parser.add_argument('--stepsize',nargs=7,required=True,type=float,dest='stepsize')
#parser.add_argument('--Nreal',required=True,type=int,dest='Nreal')
parser.add_argument('--Nsidejk',type=int,default=7,dest='Nsidejk')
parser.add_argument('--simname',required=True,dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')
parser.add_argument('--infile',required=True,dest='infile')
parser.add_argument('--outfile',required=True,dest='outfile')
#parser.add_argument('--central',type=bool,default=False,dest='central')
parser.add_argument('--parallel',type=int,default=55,dest='nproc')
args = parser.parse_args()


import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime

from halotools.sim_manager import CachedHaloCatalog

from HOD_models import decorated_hod_model
from HOD_models import standard_hod_model

from halotools.empirical_models import MockFactory

from halotools.mock_observables import return_xyz_formatted_array
from halotools.utils import randomly_downsample_data
from helpers.CorrelationFunction import projected_correlation

from Ngal_estimate import Ngal_estimate

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('ngals','wprp','wp_cov','param')

##########################################################

Lbox = args.Lbox

pi_max = 60
r_wp = np.logspace(-1, 1.4, 20)
##wp

##########################################################

def calc_all_observables(param):

    model.param_dict.update(dict(zip(param_names, param)))    ##update model.param_dict with pairs (param_names:params)
    
    n_est = c.ngal_estimate(param)
    print n_est
    if n_est>150000:
        return [0,np.zeros(19),np.zeros((19,19)),param]
    
    try:
        model.mock.populate()
    except:
        model.populate_mock(halocat)
    
    gc.collect()
    
    output = []
    
    print model.mock.galaxy_table['x'].size
    
    if model.mock.galaxy_table['x'].size<100000 or model.mock.galaxy_table['x'].size>105000:
        return [0,np.zeros(19),np.zeros((19,19)),param]
    
    pos_gals_d = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), \
            velocity=model.mock.galaxy_table['vz'], velocity_distortion_dimension='z',\
                                          period=Lbox)             ##redshift space distorted
    pos_gals_d = np.array(pos_gals_d,dtype=float)
    
        
       
    # ngals
    output.append(model.mock.galaxy_table['x'].size)
    
    
    # wprp and cov
    wp_wpcov = projected_correlation(pos_gals_d, r_wp, pi_max, Lbox, jackknife_nside=args.Nsidejk)
    output.append(wp_wpcov[0])
    output.append(wp_wpcov[1])

    
    # parameter set
    output.append(param)
    
    return output


############################################################
consuelo20_box_list = ['0_4001','0_4002','0_4003','0_4004','0_4020','0_4026','0_4027','0_4028','0_4029','0_4030',\
            '0_4032','0_4033','0_4034','0_4035','0_4036','0_4037','0_4038','0_4039','0_4040']


def main(model_gen_func, params_fname, params_usecols, output_fname):
    global model
    model = model_gen_func()

    median_w = np.median(np.loadtxt(params_fname, usecols=params_usecols),axis=0)
    params = median_w*np.ones((500+7*1000,7))  ##take medians
    
    dp_range = np.array((0.5,0.5,0.25,0.5,0.5,1,1))

    for i in params_usecols:
        params[1000*i+500:1000*i+1500,i] += (2.*np.random.random(1000)-1)*dp_range[i]

    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    global c
    
    with Pool(nproc) as pool:
        if args.simname=='consuelo20' and args.version=='all':
            for box in consuelo20_box_list:
                halocat = CachedHaloCatalog(simname = args.simname, version_name = box,redshift = args.redshift, \
                                halo_finder = args.halofinder)
                model.populate_mock(halocat)
                c = Ngal_estimate(halocat,150000)
                for i, output_data in enumerate(pool.map(calc_all_observables, params)):
                    if i%nproc == nproc-1:
                        print i
                        print str(datetime.now())
                    for name, data in zip(output_names, output_data):
                        output_dict[name].append(data)
                print box
        else:
            halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version,redshift = args.redshift, \
                                halo_finder = args.halofinder)
            model.populate_mock(halocat)
            c = Ngal_estimate(halocat,150000)
            for i, output_data in enumerate(pool.map(calc_all_observables, params)):
                if i%nproc == nproc-1:
                    print i
                    print str(datetime.now())
                for name, data in zip(output_names, output_data):
                    output_dict[name].append(data)
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(decorated_hod_model, args.infile, range(7), args.outfile+'_Fisher')
    print 'with AB done'
    f = open(args.outfile+'_log','w')
    for arg in vars(args):
        f.write(str(arg)+':'+str(getattr(args, arg))+'\n')

    f.close()


