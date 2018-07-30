import numpy as np
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models.assembias_models.heaviside_assembias import HeavisideAssembias
from halotools.utils.array_utils import custom_len
from halotools.custom_exceptions import HalotoolsError

# fix poisson distribution rvs so that it preserves ranks when a seed is specified
from halotools.empirical_models.occupation_models.occupation_model_template import OccupationComponent
from scipy.special import pdtrik

def _poisson_distribution_corrected(self, first_occupation_moment, seed=None, **kwargs):
    with NumpyRNGContext(seed):
        result = np.ceil(pdtrik(np.random.rand(*first_occupation_moment.shape),
                                first_occupation_moment)).astype(np.int)
    if 'table' in kwargs:
        kwargs['table']['halo_num_'+self.gal_type] = result
    return result

OccupationComponent._poisson_distribution = _poisson_distribution_corrected


class PreservingNgalHeavisideAssembias(HeavisideAssembias):
    def assembias_mc_occupation(self, seed=None, **kwargs):
        first_occupation_moment_orig = self.mean_occupation_orig(**kwargs)
        first_occupation_moment = self.mean_occupation(**kwargs)
        if self._upper_occupation_bound == 1:
            with NumpyRNGContext(seed):
                score = np.random.rand(custom_len(first_occupation_moment_orig))
            total = np.count_nonzero(first_occupation_moment_orig > score)
            result = np.where(first_occupation_moment > score, 1, 0)
            diff = result.sum() - total
            if diff < 0:
                x = (first_occupation_moment / score)
                result.fill(0)
                result[x.argsort()[-total:]] = 1
            elif diff > 0:
                x = (1.0-first_occupation_moment) / (1.0-score)
                result.fill(0)
                result[x.argsort()[:total]] = 1
        elif self._upper_occupation_bound == float("inf"):
            total = self._poisson_distribution(first_occupation_moment_orig.sum(), seed=seed)
            if seed is not None:
                seed += 1
            with NumpyRNGContext(seed):
                score = np.random.rand(total)
            score.sort()
            x = first_occupation_moment.cumsum(dtype=np.float64)
            x /= x[-1]
            result = np.ediff1d(np.insert(np.searchsorted(score, x), 0, 0))
        else:
            msg = ("\nYou have chosen to set ``_upper_occupation_bound`` to some value \n"
                "besides 1 or infinity. In such cases, you must also \n"
                "write your own ``mc_occupation`` method that overrides the method in the \n"
                "OccupationComponent super-class\n")
            raise HalotoolsError(msg)

        if 'table' in kwargs:
            kwargs['table']['halo_num_'+self.gal_type] = result
        return result

    def _decorate_baseline_method(self):
        self.mean_occupation_orig = self.mean_occupation
        self.mc_occupation = self.assembias_mc_occupation
        super(PreservingNgalHeavisideAssembias, self)._decorate_baseline_method()


from halotools.empirical_models import Zheng07Cens, Zheng07Sats
from halotools.empirical_models import TrivialPhaseSpace
from halotools.empirical_models import NFWPhaseSpace
from halotools.empirical_models import HodModelFactory

class PreservingNgalAssembiasZheng07Cens(Zheng07Cens, PreservingNgalHeavisideAssembias):
    def __init__(self, **kwargs):
        Zheng07Cens.__init__(self, **kwargs)
        PreservingNgalHeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound, 
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)
        
class PreservingNgalAssembiasZheng07Sats(Zheng07Sats, PreservingNgalHeavisideAssembias):
    def __init__(self, **kwargs):
        Zheng07Sats.__init__(self, **kwargs)
        PreservingNgalHeavisideAssembias.__init__(self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate='mean_occupation', **kwargs)
        
        
def decorated_hod_model():
    cen_occ_model = PreservingNgalAssembiasZheng07Cens(prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
    cen_prof_model = TrivialPhaseSpace()
    sat_occ_model = PreservingNgalAssembiasZheng07Sats(prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
    sat_prof_model = NFWPhaseSpace()
    return HodModelFactory(centrals_occupation=cen_occ_model, centrals_profile=cen_prof_model, satellites_occupation=sat_occ_model, satellites_profile=sat_prof_model)

