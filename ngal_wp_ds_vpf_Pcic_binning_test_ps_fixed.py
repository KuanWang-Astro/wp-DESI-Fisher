import sys
import argparse

parser = argparse.ArgumentParser(description='#')

parser.add_argument('--Lbox',type=int,required=True,dest='Lbox')
parser.add_argument('--simname',required=True,dest='simname')
parser.add_argument('--version',default='halotools_v0p4',dest='version')
parser.add_argument('--redshift',type=float,default=0.,dest='redshift')
parser.add_argument('--halofinder',default='rockstar',dest='halofinder')

parser.add_argument('--Nparam',type=int,required=True,dest='Nparam')
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


from halotools.mock_observables import void_prob_func
from halotools.mock_observables import wp
from halotools.mock_observables import counts_in_cylinders
from halotools.mock_observables import delta_sigma

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('func_all_1','func_all_2','func_all_3','func_all_4','func_all_5','func_all_6','param','i')  ##ngal(1)+wp{6+9+14+19+24+29}+ggl{7+10+15+20+25+30}+vpf{7+10+15+20+25+30}+Pcic{10+15+25+40+55+70}

##########################################################

Lbox = args.Lbox

pi_max = 60
r_wp_7 = np.logspace(-1, 1.5, 7)
r_wp_10 = np.logspace(-1, 1.5, 10)
r_wp_15 = np.logspace(-1, 1.5, 15)
r_wp_20 = np.logspace(-1, 1.5, 20)
r_wp_25 = np.logspace(-1, 1.5, 25)
r_wp_30 = np.logspace(-1, 1.5, 30)

##wp

r_vpf_7 = np.logspace(0,1.,7)
r_vpf_10 = np.logspace(0,1.,10)
r_vpf_15 = np.logspace(0,1.,15)
r_vpf_20 = np.logspace(0,1.,20)
r_vpf_25 = np.logspace(0,1.,25)
r_vpf_30 = np.logspace(0,1.,30)
num_sphere = int(1e5)
vpf_centers = np.loadtxt(args.vpfcen)
##vpf

proj_search_radius = 2.0         ##a cylinder of radius 2 Mpc/h
cylinder_half_length = 10.0      ##half-length 10 Mpc/h
sum_10 = [0,2,4,6,8,10,15,20,30,50]
sum_15 = [0,1,2,3,4,5,6,7,8,9,10,15,20,30,50]
sum_25 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,20,23,26,30,35,40,45,50,60]
sum_40 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,35,40,45,50,55,60,65]
sum_55 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,\
          40,41,42,43,44,45,46,47,48,50,52,55,57,60,65]
##cic

ptclpos = np.loadtxt(args.ptclpos)
rp_bins_ggl_7 = np.logspace(-1, 1.5, 8)
rp_bins_ggl_10 = np.logspace(-1, 1.5, 11)
rp_bins_ggl_15 = np.logspace(-1, 1.5, 16)
rp_bins_ggl_20 = np.logspace(-1, 1.5, 21)
rp_bins_ggl_25 = np.logspace(-1, 1.5, 26)
rp_bins_ggl_30 = np.logspace(-1, 1.5, 31)

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
    particle_masses = halocat.particle_mass
    total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    
    vpf_7 = void_prob_func(pos_gals_d, r_vpf_7, random_sphere_centers=vpf_centers, period=Lbox)
    vpf_10 = void_prob_func(pos_gals_d, r_vpf_10, random_sphere_centers=vpf_centers, period=Lbox)
    vpf_15 = void_prob_func(pos_gals_d, r_vpf_15, random_sphere_centers=vpf_centers, period=Lbox)
    vpf_20 = void_prob_func(pos_gals_d, r_vpf_20, random_sphere_centers=vpf_centers, period=Lbox)
    vpf_25 = void_prob_func(pos_gals_d, r_vpf_25, random_sphere_centers=vpf_centers, period=Lbox)
    vpf_30 = void_prob_func(pos_gals_d, r_vpf_30, random_sphere_centers=vpf_centers, period=Lbox)
    
    wprp_7 = wp(pos_gals_d, r_wp_7, pi_max, period=Lbox)
    wprp_10 = wp(pos_gals_d, r_wp_10, pi_max, period=Lbox)
    wprp_15 = wp(pos_gals_d, r_wp_15, pi_max, period=Lbox)
    wprp_20 = wp(pos_gals_d, r_wp_20, pi_max, period=Lbox)
    wprp_25 = wp(pos_gals_d, r_wp_25, pi_max, period=Lbox)
    wprp_30 = wp(pos_gals_d, r_wp_30, pi_max, period=Lbox)
    
    ggl_7 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_7, period=Lbox)[1]/1e12
    ggl_10 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_10, period=Lbox)[1]/1e12
    ggl_15 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_15, period=Lbox)[1]/1e12
    ggl_20 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_20, period=Lbox)[1]/1e12    
    ggl_25 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_25, period=Lbox)[1]/1e12
    ggl_30 = delta_sigma(pos_gals, ptclpos, particle_masses=particle_masses, downsampling_factor=downsampling_factor,\
                      rp_bins=rp_bins_ggl_30, period=Lbox)[1]/1e12
        
    Pcic_70 = np.bincount(counts_in_cylinders(pos_gals_d, pos_gals_d, proj_search_radius, \
            cylinder_half_length, period=Lbox), minlength=100)[1:71]/float(pos_gals_d.shape[0])
    Pcic_10 = np.add.reduceat(Pcic_70,sum_10)
    Pcic_15 = np.add.reduceat(Pcic_70,sum_15)
    Pcic_25 = np.add.reduceat(Pcic_70,sum_25)
    Pcic_40 = np.add.reduceat(Pcic_70,sum_40)
    Pcic_55 = np.add.reduceat(Pcic_70,sum_55)
    
    func1 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_7, ggl_7, vpf_7, Pcic_10))
    func2 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_10, ggl_10, vpf_10, Pcic_15))
    func3 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_15, ggl_15, vpf_15, Pcic_25))
    func4 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_20, ggl_20, vpf_20, Pcic_40))
    func5 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_25, ggl_25, vpf_25, Pcic_55))
    func6 = np.concatenate((np.array((pos_gals_d.shape[0]/float(Lbox**3),)), wprp_30, ggl_30, vpf_30, Pcic_70))
    
    output.append(func1)
    output.append(func2)
    output.append(func3)
    output.append(func4)
    output.append(func5)
    output.append(func6)

    
    # parameter set
    output.append(param)
    
    
    output.append(np.where(param-fid!=0)[0][0])
    
    return output


############################################################
consuelo20_box_list = ['0_4001','0_4002','0_4003','0_4004','0_4020','0_4026','0_4027','0_4028','0_4029','0_4030',\
            '0_4032','0_4033','0_4034','0_4035','0_4036','0_4037','0_4038','0_4039','0_4040']


def main(model_gen_func, fiducial, output_fname):
    global model
    model = model_gen_func()
    global fid
    fid = np.array(fiducial)
    params = fid*np.ones((7*args.Nparam,7))
    
    dp_range = np.array((0.025,0.05,0.02,0.1,0.02,0.05,0.05))

    for i in range(7):
        params[args.Nparam*i:args.Nparam*i+args.Nparam/2,i] -= dp_range[i]
        params[args.Nparam*i+args.Nparam/2:args.Nparam*i+args.Nparam,i] += dp_range[i]

    
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
    main(decorated_hod_model, args.fiducial, args.outfile)
    with open(args.outfile+'_log','w') as f:
        f.write(sys.argv[0]+'\n')
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')


