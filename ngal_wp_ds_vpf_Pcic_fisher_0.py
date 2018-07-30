import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=int,required=True,dest='Lbox')
parser.add_argument('--simname',required=True,dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--Nparam',type=int,required=True,dest='Nparam')
parser.add_argument('--Nsidejk',type=int,required=True,dest='Nsidejk')
parser.add_argument('--vpfcen',required=True,dest='vpfcen')
parser.add_argument('--ptclpos',required=True,dest='ptclpos')
parser.add_argument('--fiducial',nargs=7,required=True,type=float,dest='fiducial')
parser.add_argument('--outfile',required=True,dest='outfile')
parser.add_argument('--parallel',type=int,default=55,dest='nproc')
args = parser.parse_args()


import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime

from halotools.sim_manager import CachedHaloCatalog

from HOD_models import decorated_hod_model

from halotools.empirical_models import MockFactory

from halotools.mock_observables import return_xyz_formatted_array
from helpers.CorrelationFunction import ngal_wp_ds_vpf_Pcic_jk




##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('func_all','func_all_cov','param')  ##ngal(1)+wp(19)+ggl(20)+vpf(15)+Pcic(70)

##########################################################

Lbox = args.Lbox

pi_max = 60
r_wp = np.logspace(-1, 1.5, 20)
##wp

r_vpf = np.logspace(0,1.,15)
num_sphere = int(1e5)
vpf_centers = np.loadtxt(args.vpfcen)
##vpf

proj_search_radius = 2.0         ##a cylinder of radius 2 Mpc/h
cylinder_half_length = 10.0      ##half-length 10 Mpc/h
##cic

ptclpos = np.loadtxt(args.ptclpos)
rp_bins_ggl = np.logspace(-1, 1.5, 21)
num_ptcls_to_use = len(ptclpos)
##ggl

##########################################################

def calc_all_observables(param):

    model.param_dict.update(dict(zip(param_names, param)))    ##update model.param_dict with pairs (param_names:params)
    
    try:
        model.mock.populate()
    except:
        model.populate_mock(halocat)
    
    gc.collect()
    
    output = []
    
    pos_gals_d = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), \
            velocity=model.mock.galaxy_table['vz'], velocity_distortion_dimension='z',\
                                          period=Lbox)             ##redshift space distorted
    pos_gals_d = np.array(pos_gals_d,dtype=float)
       
    pos_gals = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), period=Lbox)
    pos_gals = np.array(pos_gals,dtype=float)
    total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    particle_masses = halocat.particle_mass*downsampling_factor
    
    
    func,funccov = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp, pi_max, r_vpf, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl, Lbox, args.Nsidejk)

    output.append(func)
    output.append(funccov)

    
    # parameter set
    output.append(param)
    
    return output


############################################################
consuelo20_box_list = ['0_4001','0_4002','0_4003','0_4004','0_4020','0_4026','0_4027','0_4028','0_4029','0_4030',\
            '0_4032','0_4033','0_4034','0_4035','0_4036','0_4037','0_4038','0_4039','0_4040']


def main(model_gen_func, fiducial, output_fname):
    global model
    model = model_gen_func()

    params = np.array(fiducial)*np.ones((args.Nparam,7))
    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    
    with Pool(nproc) as pool:
        if args.simname=='consuelo20' and args.version=='all':
            for box in consuelo20_box_list:
                halocat = CachedHaloCatalog(simname = args.simname, version_name = box,redshift = args.redshift, \
                                halo_finder = args.halofinder)
                model.populate_mock(halocat)
                for i, output_data in enumerate(pool.map(calc_all_observables, params)):
                    if 1:
                        print i
                        print str(datetime.now())
                    for name, data in zip(output_names, output_data):
                        output_dict[name].append(data)
                print box
        else:
            halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version,redshift = args.redshift, \
                                halo_finder = args.halofinder)
            model.populate_mock(halocat)
            for i, output_data in enumerate(pool.map(calc_all_observables, params)):
                if 1:
                    print i
                    print str(datetime.now())
                for name, data in zip(output_names, output_data):
                    output_dict[name].append(data)
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(decorated_hod_model, args.fiducial, args.outfile)
    with open(args.outfile+'_log','w') as f:
        f.write(sys.argv[0]+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')

