import numpy as np
import mpmath as mp
import scipy.special as sc
from numpy import linalg as LA


class Crossing1D:
    """
    Computes crossing equation for 1d bootstrap equations.

    Parameters
    ----------
    block_list : array_like
        A NumPy array containing the pregenerated conformal block data.
    params : Parameters1D
        Instance of `Parameters1D` class.
    z_data : ZData
        Instance of `ZData` class.
    """

    def __init__(self, block_list, params, z_data):
        
        self.action_space_N = params.action_space_N

        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment

        self.env_shape = z_data.env_shape
        self.z = z_data.z
        self.z_conj = z_data.z_conj

        self.use_scipy_for_hypers = True  # compute the hypergeometrics using scipy package or mpmath
        self.block_list = block_list

    def long_cons(self, delta):
        """
        Computes the (pregenerated) contribution to the crossing equation of a single block for all
        points in the z-sample simultaneously.

        Parameters
        ----------
        delta : float
            The conformal weight of the long multiplet.
        Returns
        -------
        ndarray(env_shape,)
        """

        # we clip delta to ensure that it lies within the range accessible in the pregenerated blocks
        delta = np.clip(delta, a_min=None, a_max=self.delta_start + self.delta_end_increment - self.delta_sep)
        # used a_min=None in delta because a lower bound is enforced by choosing shifts_deltas in the parameters.py file
        # now we find the nearest lattice point delta corresponds to
        n = np.rint((delta - self.delta_start) / self.delta_sep)
        # get the appropriate contribution from block_list based on spin and lattice point
        long_c = self.block_list[int(n)]
        # we need to transpose to return a shape compatible with the short multiplet contributions
        return np.transpose(long_c)

    def long_coeffs_array(self, deltas):
        """
        Aggregates all the long multiplet contributions together into a single array.

        Returns
        -------
        long_c : ndarray(num_of_long, env_shape)
        """
        long_c = self.long_cons(deltas[0])
        for x in range(1, deltas.size):
            long_c = np.vstack((long_c, self.long_cons(deltas[x])))
        return long_c


class Crossing1D_SAC(Crossing1D):

    def __init__(self, block_list, params, z_data):
        super().__init__(block_list, params, z_data)

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.dyn_shift = params.dyn_shift  # the weight separation value
        #self.dup_list = self.spin_list_long == np.roll(self.spin_list_long, -1)  # which long spins are degenerate

    def split_cft_data(self, cft_data):
        """
        Sets up dictionaries to decompose the search space data into easily identifiable pieces.

        Parameters
        ----------
        cft_data : ndarray
            The array to be split.

        Returns
        -------
        delta_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the conformal weights.
        ope_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the OPE-squared coefficients.

        """
        delta_dict = {
            "long": cft_data[self.multiplet_index[0]]
        }
        ope_dict = {
            "long": cft_data[self.multiplet_index[0] + self.action_space_N // 2]
        }
        return delta_dict, ope_dict

    def impose_weight_separation(self, delta_dict):
        """
        Enforces a minimum conformal dimension separation between long multiplets of the same spin by
        overwriting values of delta_dict.

        Parameters
        ----------
        delta_dict : dict
            A dictionary of multiplet types and their conformal weights.
        Returns
        -------
        delta_dict : dict
            Dictionary with modified values for 'long' key.
        """
        deltas = delta_dict['long']
        flag_current = False
        flag_next = False
        for i in range(self.dup_list.size):
            flag_current = self.dup_list[i]
            flag_next_tmp = False

            if flag_next and not flag_current:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)

            if flag_current and not flag_next:
                flag_next_tmp = True

            if flag_current and flag_next:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)
                flag_next_tmp = True

            flag_next = flag_next_tmp

        return delta_dict

    def crossing(self, cft_data):
        """
        Evaluates the truncated crossing equations for the given CFT data at all points in the z-sample simultaneously.

        Parameters
        ----------
        cft_data : ndarray
            An array containing the conformal weights and OPE-squared coefficients of all the multiplets.

        Returns
        -------
        constraints : ndarray
            Array of values of the truncated crossing equation.
        reward : float
            The reward determined from the constraints.
        cft_data : ndarray
            A list of possibly modified CFT data.

        """
        # get some dictionaries
        delta_dict, ope_dict = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            # impose the mimimum conformal weight separations between operators
            delta_dict = self.impose_weight_separation(delta_dict)
            # since we've altered some data we update the long multiplet weights in cft_data
            cft_data[self.multiplet_index[2]] = delta_dict['long']

        # broadcast the reshaped long multiplet ope coefficients over their crossing contributions
        long_cons = ope_dict['long'].reshape(-1, 1) * self.long_coeffs_array(delta_dict['long'])
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        constraints = self.INHO_VALUE + long_cons.sum(axis=0)  
        # the .sum implements summation over multiplet spins
        reward = 1 / LA.norm(constraints)

        return constraints, reward, cft_data