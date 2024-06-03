from .mh import MHMove
from .gaussian import reflect_cosines_array

import numpy as np
import warnings

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
            **kwargs: Additional keyword arguments for parent class.
        
        """
    
    def __init__(self, chain_container=None, factor=None, sky_periodic=None, F=0.5, CR=0.9, use_current_state=True, crossover=True, **kwargs):
    
        self.chain_container = chain_container
        if self.chain_container is None:
            warnings.warn("No chain container provided. The current state will be used for the proposal. This will not satisfy detailed balance.")

        self.sky_periodic = sky_periodic

        self.F = F
        self.CR = CR
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
        
        if self.use_current_state:
            mutant_vectors = current_state + F * (chain[indices[:, 1]] - chain[indices[:, 2]])
            # Add a small random number to each parameter
            epsilon = mutant_vectors * 1e-4
            mutant_vectors += epsilon * np.random.randn(n_walkers, n_params)
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
    
    def get_updated_vector(self, rng, x0, chain):
        
        prob = rng.random()

        if prob > 0.5:
            F = self.get_factor(rng)
            CR = np.random.uniform(0.5, 1.0)

        else:
            F = self.F
            CR = self.CR
        
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
                chain = coords

            else:
                if self.chain_container.sampler.iteration > 0:
                    self.chain_container.update_chain(T=slice(0, ntemps), reshape=False)
                    chain = self.chain_container.chain[name]
                    # randomly select a previous step from the chain
                    step = random.randint(0, chain.shape[0])
                    chain = chain[step]
                
                else: 
                    chain = coords

            # get new points
            proposed_coords = self.get_updated_vector(random, coords[inds_here], chain[inds_here])

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