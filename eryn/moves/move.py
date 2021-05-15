# -*- coding: utf-8 -*-

import numpy as np

__all__ = ["Move"]


class Move(object):
    def __init__(self, temperature_control=None):
        self.temperature_control = temperature_control

        if self.temperature_control is None:
            self.compute_log_posterior = self.compute_log_posterior_basic
        else:
            self.compute_log_posterior = (
                self.temperature_control.compute_log_posterior_tempered
            )

    def compute_log_posterior_basic(self, logl, logp):
        return logl + logp

    def tune(self, state, accepted):
        pass

    def update(self, old_state, new_state, accepted, subset=None):
        """Update a given subset of the ensemble with an accepted proposal

        Args:
            coords: The original ensemble coordinates.
            log_probs: The original log probabilities of the walkers.
            blobs: The original blobs.
            new_coords: The proposed coordinates.
            new_log_probs: The proposed log probabilities.
            new_blobs: The proposed blobs.
            accepted: A vector of booleans indicating which walkers were
                accepted.
            subset (Optional): A boolean mask indicating which walkers were
                included in the subset. This can be used, for example, when
                updating only the primary ensemble in a :class:`RedBlueMove`.

        """
        if subset is None:
            subset = np.tile(
                np.arange(old_state.log_prob.shape[1]), (old_state.log_prob.shape[0], 1)
            )

        accepted_temp = np.take_along_axis(accepted, subset, axis=1)

        old_log_probs = np.take_along_axis(old_state.log_prob, subset, axis=1)
        new_log_probs = new_state.log_prob
        temp_change_log_prob = new_log_probs * (accepted_temp) + old_log_probs * (
            ~accepted_temp
        )

        np.put_along_axis(old_state.log_prob, subset, temp_change_log_prob, axis=1)

        old_log_priors = np.take_along_axis(old_state.log_prior, subset, axis=1)
        new_log_priors = new_state.log_prior.copy()

        # deal with -infs
        new_log_priors[np.isinf(new_log_priors)] = 0.0

        temp_change_log_prior = new_log_priors * (accepted_temp) + old_log_priors * (
            ~accepted_temp
        )

        np.put_along_axis(old_state.log_prior, subset, temp_change_log_prior, axis=1)

        # inds
        old_inds = {
            name: np.take_along_axis(branch.inds, subset[:, :, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_inds = {name: branch.inds for name, branch in new_state.branches.items()}

        temp_change_inds = {
            name: new_inds[name] * (accepted_temp[:, :, None])
            + old_inds[name] * (~accepted_temp[:, :, None])
            for name in old_inds
        }

        [
            np.put_along_axis(
                old_state.branches[name].inds,
                subset[:, :, None],
                temp_change_inds[name],
                axis=1,
            )
            for name in new_inds
        ]

        # coords
        old_coords = {
            name: np.take_along_axis(branch.coords, subset[:, :, None, None], axis=1)
            for name, branch in old_state.branches.items()
        }

        new_coords = {
            name: branch.coords for name, branch in new_state.branches.items()
        }

        temp_change_coords = {
            name: new_coords[name] * (accepted_temp[:, :, None, None])
            + old_coords[name] * (~accepted_temp[:, :, None, None])
            for name in old_coords
        }

        [
            np.put_along_axis(
                old_state.branches[name].coords,
                subset[:, :, None, None],
                temp_change_coords[name],
                axis=1,
            )
            for name in new_coords
        ]

        if new_state.blobs is not None:
            raise NotImplementedError
            if old_state.blobs is None:
                raise ValueError(
                    "If you start sampling with a given log_prob, "
                    "you also need to provide the current list of "
                    "blobs at that position."
                )
            old_state.blobs[m1] = new_state.blobs[m2]

        return old_state