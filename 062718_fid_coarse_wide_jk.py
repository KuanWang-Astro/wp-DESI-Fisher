import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=float,default=250.,dest='Lbox')
parser.add_argument('--simname',default='bolplanck',dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--Nsidejk',type=int,required=True,dest='Nsidejk')

parser.add_argument('--Nparam',type=int,required=True,dest='Nparam')
parser.add_argument('--vpfcen',required=True,dest='vpfcen')
parser.add_argument('--ptclpos',required=True,dest='ptclpos')
#parser.add_argument('--fiducial',nargs=7,required=True,type=float,dest='fiducial')
parser.add_argument('--threshold',required=True,type=float,dest='threshold')
parser.add_argument('--outfile',required=True,dest='outfile')
parser.add_argument('--parallel',type=int,default=55,dest='nproc')
args = parser.parse_args()


import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime
print str(datetime.now())

from halotools.sim_manager import CachedHaloCatalog

from HOD_models_fix import decorated_hod_model

from halotools.empirical_models import MockFactory

from halotools.mock_observables import return_xyz_formatted_array
from helpers.CorrelationFunction import ngal_wp_ds_vpf_Pcic_jk




##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('func_all','func_all_cov','param')  ##ngal(1)+wp(19)+ggl(20)+vpf(15)+Pcic(70)

##########################################################

p19p0 = np.array((1.04029, 12.80315, 0.51193, 10.25010, 11.64354, 0., 0.))
p19p5 = np.array((1.11553, 13.06008, 0.44578, 11.29134, 11.75068, 0., 0.))
p20p0 = np.array((1.14385, 13.28584, 0.34846, 11.30750, 11.97186, 0., 0.))
p20p5 = np.array((1.19652, 13.59169, 0.18536, 11.20134, 12.25470, 0., 0.))
p21p0 = np.array((1.33738, 13.98811, 0.55950, 11.95797, 12.82356, 0., 0.))

if args.threshold==-19.0:
    fiducial_p = p19p0
elif args.threshold==-19.5:
    fiducial_p = p19p5
elif args.threshold==-20.0:
    fiducial_p = p20p0
elif args.threshold==-20.5:
    fiducial_p = p20p5
elif args.threshold==-21.0:
    fiducial_p = p21p0
    
#########################################################


Lbox = args.Lbox

pi_max = 60
r_wp = np.logspace(-1, 1.5, 20)
##wp

r_vpf = np.logspace(0,1.,20)
num_sphere = int(1e5)
vpf_centers = np.loadtxt(args.vpfcen)
##vpf

proj_search_radius = 2.0         ##a cylinder of radius 2 Mpc/h
cylinder_half_length = 10.0      ##half-length 10 Mpc/h

sum_40 = np.arange(40)
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
    
    
    func,funccov = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp, pi_max, r_vpf, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl, Lbox, args.Nsidejk, sum_40)
    
    output.append(func)
    output.append(funccov)

    
    # parameter set
    output.append(param)
    
    return output


############################################################

def main(model_gen_func, fiducial, output_fname):
    global model
    model = model_gen_func()

    params = np.array(fiducial)*np.ones((args.Nparam,7))
    
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    
    global halocat
    
    with Pool(nproc) as pool:
        if 1:
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
    main(decorated_hod_model, fiducial_p, args.outfile)
    with open(args.outfile+'_log','w') as f:
        f.write(sys.argv[0]+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')

