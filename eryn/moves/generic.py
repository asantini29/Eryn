import numpy as np
from dataclasses import dataclass
from typing import Callable
import warnings

from .move import Move
from .mh import MHMove

__all__ = ["ChainContainer", "Proposal"]

"""
A generic class interface for different types of proposals. 
The content of this module is heavily inspired by the `impulse_mcmc` sampler (https://github.com/AaronDJohnson/impulse_mcmc/blob/main).
"""

@dataclass
class ChainContainer:
    """
    A class that represents a container for storing and updating Markov chain samples.

    Attributes:
        sampler (object): The sampler object used to generate the chain.
        chain (np.ndarray): The Markov chain samples.
        chain_cov (np.ndarray): The covariance matrix of the chain.
        chain_mean (np.ndarray): The mean of the chain.
        buffer_size (int): The maximum size of the chain buffer.

    Methods:
        __post_init__(): Initializes the chain container and updates the covariance and mean if necessary.
        update_cov(): Updates the covariance matrix of the chain.
        update_mean(): Updates the mean of the chain.
        update_chain(T=0): Updates the chain by retrieving samples from the sampler.
        update(T=0): Updates the chain container by updating the chain, covariance, and mean.

    """

    sampler: object = None
    chain: np.ndarray = None
    chain_cov: np.ndarray = None
    chain_mean: np.ndarray = None
    buffer_size: int = 1000

    def __post_init__(self):
        """
        Initializes the chain container and updates the covariance and mean if necessary.
        """
        if self.sampler is not None:
            self.branches = self.sampler.branch_names

        if self.chain is not None:
            if self.chain_cov is None:
                self.update_cov()
            if self.chain_mean is None:
                self.update_mean()

    def update_cov(self):
        """
        Updates the covariance matrix of the chain.
        """
        chain_cov = {}
        chain_svd = {}
        for name, chain in self.chain.items():
            cov = np.cov(chain, rowvar=False)
            chain_cov[name] = cov

            u, s, v = np.linalg.svd(cov)
            chain_svd[name] = (u, s, v)
        
        self.chain_cov = chain_cov
        self.chain_svd = chain_svd 

    def update_mean(self):
        """
        Updates the mean of the chain.
        """
        chain_mean = {}
        for name, chain in self.chain.items():
            chain_mean[name] = np.mean(chain, axis=0)
        
        self.chain_mean = chain_mean

    def update_chain(self, new_chain=None, T=0):
        """
        Updates the chain by retrieving samples from the sampler.

        Args:
            T (int): The index of the temperature to use for the update (default is 0).
        """

        if self.sampler is None:
            if new_chain is not None:
                self.chain = new_chain
            else:
                warnings.warn("Neither sampler or new chain provided to update the current chain.")
        else:        
            self.chain = self.sampler.get_chain()[:self.buffer_size, T]

    def update(self, T=0):
        """
        Updates the chain container by updating the chain, covariance, and mean.
        """
        self.update_chain(T=T)
        self.update_cov()
        self.update_mean()

class CustomProposal(MHMove):
    """
    Generic proposal interface.
    """
    def __init__(self, 
                 all_proposals: dict[str, Callable] = None,               
                 chain_container: ChainContainer = None,
                 **kwargs
                 ):
        
        self.all_proposals = all_proposals
        self.sanity_check()
        
        self.chain_container = chain_container

        super(CustomProposal, self).__init__(**kwargs)


    def sanity_check(self):
            """
            Perform a sanity check on the proposals.

            Raises:
                ValueError: If any proposal is not callable.
            """
            for name, proposal in self.all_proposals.items():
                if not isinstance(proposal, Callable):
                    raise ValueError(f"Proposal {name} is not callable.")

        

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        """
        Get proposal from Gaussian distribution

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

        self.chain_container.update()

        q = {}

        for name, coords in zip(branches_coords.keys(), branches_coords.values()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape

            # setup inds accordingly
            if branches_inds is None:
                inds = np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
            else:
                inds = branches_inds[name]

            # get the proposal for this branch
            proposal_fn = self.all_proposal[name]
            inds_here = np.where(inds == True)

            # copy coords
            q[name] = coords.copy()

            # get new points
            new_coords, factors = proposal_fn(coords[inds_here], random)

            q[name][inds_here] = new_coords.copy()

        return q, factors
