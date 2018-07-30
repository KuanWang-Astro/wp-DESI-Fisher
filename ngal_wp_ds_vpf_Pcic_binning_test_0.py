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
output_names = ('func_all_1','func_all_cov_1','func_all_2','func_all_cov_2',\
                'func_all_3','func_all_cov_3','func_all_4','func_all_cov_4',\
                'func_all_5','func_all_cov_5','func_all_6','func_all_cov_6','param')  ##ngal(1)+wp{6+9+14+19+24+29}+ggl{7+10+15+20+25+30}+vpf{7+10+15+20+25+30}+Pcic{10+15+25+40+55+70}

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
    total_num_ptcls_in_snapshot = halocat.num_ptcl_per_dim**3
    downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
    particle_masses = halocat.particle_mass*downsampling_factor
    
    
    func1,funccov1 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_7, pi_max, r_vpf_7, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_7, Lbox, args.Nsidejk, sum_10)
    func2,funccov2 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_10, pi_max, r_vpf_10, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_10, Lbox, args.Nsidejk, sum_15)
    func3,funccov3 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_15, pi_max, r_vpf_15, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_15, Lbox, args.Nsidejk, sum_25)
    func4,funccov4 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_20, pi_max, r_vpf_20, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_20, Lbox, args.Nsidejk, sum_40)
    func5,funccov5 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_25, pi_max, r_vpf_25, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_25, Lbox, args.Nsidejk, sum_55)
    func6,funccov6 = ngal_wp_ds_vpf_Pcic_jk(pos_gals_d, r_wp_30, pi_max, r_vpf_30, vpf_centers, proj_search_radius, cylinder_half_length, pos_gals, ptclpos, particle_masses, rp_bins_ggl_30, Lbox, args.Nsidejk)

    output.append(func1)
    output.append(funccov1)
    output.append(func2)
    output.append(funccov2)
    output.append(func3)
    output.append(funccov3)
    output.append(func4)
    output.append(funccov4)
    output.append(func5)
    output.append(funccov5)
    output.append(func6)
    output.append(funccov6)
    
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

