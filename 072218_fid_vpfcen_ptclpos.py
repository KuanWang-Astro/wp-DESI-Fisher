import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=float,default=250.,dest='Lbox')
parser.add_argument('--simname',default='bolplanck',dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--Nvpf',type=int,required=True,dest='Nvpf')
parser.add_argument('--Nds',type=int,required=True,dest='Nds')
parser.add_argument('--vpfNbin',type=int,required=True,dest='vpfNbin')
parser.add_argument('--dsNbin',type=int,required=True,dest='dsNbin')
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
from halotools.mock_observables import delta_sigma
from halotools.mock_observables import void_prob_func

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('vpf','deltasigma')

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

r_vpf = np.logspace(0,1.,args.vpfNbin)
num_sphere = int(1e5)
##vpf

ptcl_accept_rate = 0.1
rp_bins_ggl = np.logspace(-1, 1.5, args.dsNbin+1)
##ggl

##########################################################

halocat = CachedHaloCatalog(simname = args.simname, version_name = args.version, redshift = args.redshift, \
                                halo_finder = args.halofinder)

model = decorated_hod_model()
model.param_dict.update(dict(zip(param_names, fiducial_p)))    ##update model.param_dict with pairs (param_names:params)
    
try:
    model.mock.populate()
except:
    model.populate_mock(halocat)
    
gc.collect()

pos_gals_d = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), \
            velocity=model.mock.galaxy_table['vz'], velocity_distortion_dimension='z',\
                                          period=Lbox)             ##redshift space distorted
pos_gals_d = np.array(pos_gals_d,dtype=float)
       
pos_gals = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), period=Lbox)
pos_gals = np.array(pos_gals,dtype=float)

particle_masses = halocat.particle_mass
total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3

##########################################################

def random_vpfcen(nsphere):
    return np.random.rand(num_sphere,3)*Lbox

def downsample_ptclpos(rate):
    mask = np.random.rand(len(halocat.ptcl_table))<rate
    table = halocat.ptcl_table[mask]
    return return_xyz_formatted_array(*(table[ax] for ax in 'xyz'), period=Lbox)

##########################################################

def calc_vpf(i):
    vpf_centers = random_vpfcen(num_sphere)
    return void_prob_func(pos_gals_d, r_vpf, random_sphere_centers=vpf_centers, period=Lbox)

def calc_ds(i):
    ptclpos = downsample_ptclpos(ptcl_accept_rate)
    num_ptcls_to_use = len(ptclpos)
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    return delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl, period=Lbox)[1]/1e12

##########################################################

def main():
    output_dict = collections.defaultdict(list)
    nproc = args.nproc
    with Pool(nproc) as pool:
        for i, output_data in enumerate(pool.map(calc_vpf, range(args.Nvpf))):
            if i%nproc == nproc-1:
                print i
                print str(datetime.now())
            output_dict['vpf'].append(output_data)
    with Pool(nproc) as pool:
        for i, output_data in enumerate(pool.map(calc_ds, range(args.Nds))):
            if i%nproc == nproc-1:
                print i
                print str(datetime.now())
            output_dict['deltasigma'].append(output_data)
    
    
    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(args.outfile, **output_dict)


if __name__ == '__main__':
    main()
    with open(args.outfile+'_log','w') as f:
        f.write(sys.argv[0]+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')