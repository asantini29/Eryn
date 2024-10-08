from .mh import MHMove
from .gaussian import reflect_cosines_array
from .red_blue import RedBlueMove
from .group import GroupMove

import numpy as np
import warnings

from functools import lru_cache

__all__ = ["DEMove"]

"""
A class for Differential Evolution (DE) moves that uses a ``ChainContainer`` object to propose new points.
"""

class DEMove(MHMove):
    """
        A Metropolis step with Differential Evolution proposal function..

        Args:
            chain_container (object, optional): A ``ChainContainer`` object that contains the chain, covariance, and mean (default is ``None``).
            factor (float, optional): The factor to use for the differential evolution (default is ``None``).
            sky_periodic (list, optional): A list of tuples where each tuple contains the name of the branch and the indices of the parameters that are sky periodic (default is ``None``).
            F (float, optional): The differential evolution factor (default is 0.5).
            CR (float, optional): The differential evolution crossover rate (default is 0.9).
            use_current_state (bool, optional): Whether to use the current state as the chain (default is ``False``).
            crossover (bool, optional): Whether to use crossover (default is ``True``).
            n_iter (int, optional): The number of iterations to run before updating the chain (default is 500).
            **kwargs: Additional keyword arguments for parent class.
        
        """
    
    def __init__(self, chain_container=None, factor=None, sky_periodic=None, F=0.5, CR=0.9, sigma=1e-5, g0=None, ndims=None, use_current_state=True, crossover=True, n_iter=500, **kwargs):
    
        self.chain_container = chain_container
        if self.chain_container is None:
            warnings.warn("No chain container provided. The current state will be used for the proposal. This will not satisfy detailed balance.")

        self.n_iter = n_iter
        self.sky_periodic = sky_periodic

        self.F = F
        self.CR = CR
        self.sigma = sigma

        if isinstance(g0, dict):
            self.g0 = g0

        elif isinstance(ndims, dict):
            self.g0 = {k: 2.38/np.sqrt(2*ndims[k]) for k in ndims.keys()}
        
        elif isinstance(ndims, int):
            self.g0 = 2.38/np.sqrt(2*ndims)

        else:
            raise ValueError("Either 'g0' or 'ndims' must be provided.")

        self.use_current_state = use_current_state
        if not self.use_current_state:
            warnings.warn("Not using the current state for the proposal. This will not satisfy detailed balance.")

        self.crossover = crossover

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        super().__init__(**kwargs)

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp( rng.uniform( -self._log_factor, 0.0 ) )
    
    def propose_DE(self, current_state, chain, F, CR):

        n_walkers, n_params = current_state.shape

        # Randomly select three distinct indices for each walker
        indices = np.random.choice(chain.shape[0], size=(chain.shape[0], 3), replace=True)      

        while np.any(np.diff(indices, axis=1) == 0):
            indices = np.random.choice(chain.shape[0], size=(chain.shape[0], 3), replace=True)
        
        if self.use_current_state:
            mutant_vectors = current_state + F * (chain[indices[:, 1]] - chain[indices[:, 2]])
        else:
            mutant_vectors = chain[indices[:, 0]] + F * (chain[indices[:, 1]] - chain[indices[:, 2]])


        # Perform crossover with the current state to create the proposed state
        if self.crossover:
            crossover_mask = (np.random.rand(n_walkers, n_params) <= CR) | (np.arange(n_params) == np.random.randint(n_params, size=(n_walkers, 1)))
        else:
            # to update all
            crossover_mask = np.ones((n_walkers, n_params), dtype=bool)

        proposed_state = np.where(crossover_mask, mutant_vectors, current_state)

        return proposed_state
    
    def get_updated_vector(self, rng, g0, x0, chain):
        
        prob = rng.random()

        if prob > 0.5:
            #F = self.get_factor(rng)
            CR = np.random.uniform(0.5, 1.0)

        else:
            #F = self.F
            CR = self.CR
        
        F = g0 * (1 + self.sigma * rng.normal(0, 1))

        return self.propose_DE(current_state=x0, chain=chain, F=F, CR=CR)
    
    


    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """
        Args:
            branches_coords (dict): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim] representing
                coordinates for walkers.
            random (object): Current random state object.
            branches_inds (dict, optional): Keys are ``branch_names`` and values are
                np.ndarray[ntemps, nwalkers, nleaves_max] representing which
                leaves are currently being used. (default: ``None``)
            **kwargs (ignored): This is added for compatibility. It is ignored in this function.

        Returns:
            tuple: (Proposed coordinates, factors) -> (dict, np.ndarray)

        """

        q = {}

        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            if self.chain_container is None:
                chain = coords[inds_here]

            else:
                if self.chain_container.sampler.iteration > 0 and self.chain_container.sampler.iteration % self.n_iter == 0:
                    self.chain_container.update_chain(T=slice(0, ntemps), reshape=False)

                if self.chain_container.chain is None:
                    chain = coords[inds_here]
                
                else:
                    #breakpoint()
                    chain = self.chain_container.chain[name]
                    chain_inds = self.chain_container.inds[name]
                    #* randomly select a previous step from the chain
                    step = random.randint(0, chain.shape[0])
                    chain = chain[step]
                    chain_inds = chain_inds[step]

                    #chain = chain[np.where(chain_inds)]
                    chain = chain[inds_here] # not working with reversible jump


            # get new points
            g0 = self.g0[name] if isinstance(self.g0, dict) else self.g0
            proposed_coords = self.get_updated_vector(random, g0, coords[inds_here], chain)

            new_coords = proposed_coords.copy()

            if self.sky_periodic:
                indx_list_here = [el[1] for el in self.sky_periodic if el[0]==name]
                nw = proposed_coords.shape[0]
                for temp_ind in range(len(indx_list_here)):
                    csth = proposed_coords[:,indx_list_here[temp_ind][0]][:,0]
                    ph = proposed_coords[:,indx_list_here[temp_ind][0]][:,1]
                    new_coords[:,indx_list_here[temp_ind][0]] = np.asarray(reflect_cosines_array(csth, ph)).T
                
            # put into coords in proper location
            q[name][inds_here] = new_coords.copy()

        # handle periodic parameters
        if self.periodic is not None:
            for name, tmp in q.items():
                ntemps, nwalkers, nleaves_max, ndim = tmp.shape
                q[name] = self.periodic.wrap({name: tmp.reshape(ntemps * nwalkers, nleaves_max, ndim)})
                q[name] = tmp.reshape(ntemps, nwalkers, nleaves_max, ndim)

        return q, np.zeros(shape=(ntemps, nwalkers))
    

class DEMove_emcee(RedBlueMove):
    r"""A proposal using differential evolution. This proposal is directly based on the `emcee` implementation.

    This `Differential evolution proposal
    <http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf>`_ is
    implemented following `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    Args:
        sigma (float): The standard deviation of the Gaussian used to stretch
            the proposal vector.
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default, it is :math:`2.38 / \sqrt{2\,\mathrm{ndim}}`
            as recommended by the two references.

    """

    def __init__(self, sigma=1.0e-5, gamma0=None, return_gpu=False, random_seed=None, target_acceptance=0.25, **kwargs):
        self.sigma = sigma
        self.gamma0 = gamma0
        self.target_acceptance = target_acceptance

        super().__init__(**kwargs)

        if random_seed is not None:
            self.xp.random.seed(random_seed)

        self.return_gpu = return_gpu

    def setup(self, branch_coords):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Pure MAGIC:
            self.g0 = {}
            for key, coords in branch_coords.items():
                ndim = coords.shape[-1]
                self.g0[key] = 2.38 / np.sqrt(2 * ndim)
    
    def choose_c_vals(self, c, Nc, Ns, ntemps, random_number_generator, **kwargs):
        """Get the compliment array

        The compliment represents the points that are used to move the actual points whose position is
        changing.

        Args:
            c (np.ndarray): Possible compliment values with shape ``(ntemps, Nc, nleaves_max, ndim)``.
            Nc (int): Length of the ``...``: the subset of walkers proposed to move now (usually nwalkers/2).
            Ns (int): Number of generation points.
            ntemps (int): Number of temperatures.
            random_number_generator (object): Random state object.
            **kwargs (ignored): Ignored here. For modularity.

        Returns:
            np.ndarray: Compliment values to use with shape ``(ntemps, Ns, nleaves_max, ndim)``.

        """

        rint = random_number_generator.randint(
            Nc,
            size=(
                ntemps,
                Ns,
            ),
        )
        c_temp = self.xp.take_along_axis(c, rint[:, :, None, None], axis=1)
        return c_temp
    
    def get_new_points(
        self, name, s, c_temp, Ns, g0, branch_shape, branch_i, random_number_generator
    ):
        """Get mew points in de move.

        Takes compliment and uses it to get new points for those being proposed.

        Args:
            name (str): Branch name.
            s (np.ndarray): Points to be moved with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            c_temp (np.ndarray): Compliment to move points with shape ``(ntemps, Ns, nleaves_max, ndim)``.
            Ns (int): Number to generate.
            branch_shape (tuple): Full branch shape.
            branch_i (int): Which branch in the order is being run now. This ensures that the
                randomly generated quantity per walker remains the same over branches.
            random_number_generator (object): Random state object.

        Returns:
            np.ndarray: New proposed points with shape ``(ntemps, Ns, nleaves_max, ndim)``.


        """

        ntemps, nwalkers, nleaves_max, ndim_here = branch_shape

        # get proper distance
        nc = c_temp.shape[1]
        pairs = _get_nondiagonal_pairs(nc)

        # Sample from the pairs
        indices = random_number_generator.choice(pairs.shape[0], size=Ns, replace=True)
        pairs = pairs[indices]

        #breakpoint()

        # Compute diff vectors 
        gamma = g0 * (1 + self.sigma * random_number_generator.randn(1, Ns, 1, 1))

        if self.periodic is not None:
            raise NotImplementedError("Periodic boundaries not implemented yet. have to check")
            diff = self.periodic.distance(
                {name: c_temp[:, pairs[0]].reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                {name: c_temp[:, pairs[1]].reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)
        else:
            #diff = c_temp - s
            diff = np.diff(c_temp[:, pairs], axis=2).squeeze(axis=2)

        temp =  s + gamma * (diff)

        # wrap periodic values

        if self.periodic is not None:
            temp = self.periodic.wrap(
                {name: temp.reshape(ntemps * nwalkers, nleaves_max, ndim_here)},
                xp=self.xp,
            )[name].reshape(ntemps, nwalkers, nleaves_max, ndim_here)

        # get from gpu or not
        if self.use_gpu and not self.return_gpu:
            temp = temp.get()
        return temp

    def get_proposal(self, s_all, c_all, random, gibbs_ndim=None, **kwargs):
        """Generate stretch proposal

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            c_all (dict): Keys are ``branch_names`` and values are lists. These
                lists contain all the complement array values.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """

        # needs to be set before we reach the end
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            if not isinstance(c_all[name], list):
                raise ValueError("c_all for each branch needs to be a list.")

            # get compliment possibilities
            c = [self.xp.asarray(c_tmp) for c_tmp in c_all[name]]

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape
            c = self.xp.concatenate(c, axis=1)

            Ns, Nc = s.shape[1], c.shape[1]
            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            # need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")

            # get actual compliment values
            c_temp = self.choose_c_vals(c, Nc, Ns, ntemps, random_number_generator)
            g0 = self.g0[name]
            # use stretch to get new proposals
            newpos[name] = self.get_new_points(
                name, s, c_temp, Ns, g0, s.shape, i, random_number_generator
            )
        # proper factors
        factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        return newpos, factors
    
    def tune(self, **kwargs):
        """Tune the proposal parameters.

        This method tunes the proposal parameters based on the acceptance fraction of the chain."""
        
        current_acceptance = np.mean(self.acceptance_fraction[0])
        print(f"Current acceptance fraction: {current_acceptance}")
        self.gamma0 = self.g0.copy()
        for key in self.gamma0.keys():
            if current_acceptance > 0.31:
                self.gamma0[key] *= 1.1
            elif current_acceptance < 0.2:
                self.gamma0[key] *= 0.9
            else:
                self.gamma0[key] *= np.sqrt(current_acceptance / self.target_acceptance)



@lru_cache(maxsize=1)
def _get_nondiagonal_pairs(n: int) -> np.ndarray:
    """Get the indices of a square matrix with size n, excluding the diagonal."""
    rows, cols = np.tril_indices(n, -1)  # -1 to exclude diagonal

    # Combine rows-cols and cols-rows pairs
    pairs = np.column_stack(
        [np.concatenate([rows, cols]), np.concatenate([cols, rows])]
    )

    return pairs

class GroupDEMove(GroupMove, DEMove_emcee):
    def __init__(self, **kwargs):
        GroupMove.__init__(self, **kwargs)
        DEMove_emcee.__init__(self, **kwargs)

    def setup(self, branch_coords):
        self.g0 = self.gamma0
        if self.g0 is None:
            # Pure MAGIC:
            self.g0 = {}
            for key, coords in branch_coords.items():
                ndim = coords.shape[-1]
                self.g0[key] = 2.38 / np.sqrt(2 * ndim)

    def tune(self, **kwargs):
        """Tune the proposal parameters.

        This method tunes the proposal parameters based on the acceptance fraction of the chain."""
        
        current_acceptance = np.mean(self.acceptance_fraction[0])
        print(f"Current acceptance fraction: {current_acceptance}")
        self.gamma0 = self.g0.copy()
        for key in self.gamma0.keys():
            if current_acceptance > 0.31:
                self.gamma0[key] *= 1.1
            elif current_acceptance < 0.2:
                self.gamma0[key] *= 0.9
            else:
                self.gamma0[key] *= np.sqrt(current_acceptance / self.target_acceptance)


    def get_proposal(self, s_all, random, gibbs_ndim=None, s_inds_all=None, **kwargs):
        """Generate group stretch proposal coordinates

        Args:
            s_all (dict): Keys are ``branch_names`` and values are coordinates
                for which a proposal is to be generated.
            random (object): Random state object.
            gibbs_ndim (int or np.ndarray, optional): If Gibbs sampling, this indicates
                the true dimension. If given as an array, must have shape ``(ntemps, nwalkers)``.
                See the tutorial for more information.
                (default: ``None``)
            s_inds_all (dict, optional): Keys are ``branch_names`` and values are 
                ``inds`` arrays indicating which leaves are currently used. (default: ``None``)

        Returns:
            tuple: First entry is new positions. Second entry is detailed balance factors.

        Raises:
            ValueError: Issues with dimensionality.

        """
        # needs to be set before we reach the end
        random_number_generator = random if not self.use_gpu else self.xp.random
        newpos = {}

        # iterate over branches
        for i, name in enumerate(s_all):
            # get points to move
            s = self.xp.asarray(s_all[name])

            Ns = s.shape[1]

            if s_inds_all is not None:
                s_inds = self.xp.asarray(s_inds_all[name])
            else:
                s_inds = None

            ntemps, nwalkers, nleaves_max, ndim_here = s.shape

            # gets rid of any values of exactly zero
            ndim_temp = nleaves_max * ndim_here

            # need to properly handle ndim
            if i == 0:
                ndim = ndim_temp
                Ns_check = Ns

            else:
                ndim += ndim_temp
                if Ns_check != Ns:
                    raise ValueError("Different number of walkers across models.")


            Ns = nwalkers

            # get actual compliment values
            c_temp = self.choose_c_vals(name, s, s_inds=s_inds)
            g0 = self.g0[name]
            # breakpoint()
            # use stretch to get new proposals
            newpos[name] = self.get_new_points(
                name, s, c_temp, Ns, g0, s.shape, i, random_number_generator
            )

        # proper factors
        factors = self.xp.zeros((ntemps, nwalkers), dtype=self.xp.float64)
        if self.use_gpu and not self.return_gpu:
            factors = factors.get()

        return newpos, factors