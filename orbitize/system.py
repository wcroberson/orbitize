import numpy as np
from orbitize import nbody, kepler, basis, hipparcos
from astropy import table
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from orbitize.read_input import read_file
from copy import deepcopy


class System(object):
    """
    A class to store information about a system (data & priors)
    and calculate model predictions given a set of orbital
    parameters.

    Args:
        num_secondary_bodies (int): number of secondary bodies in the system.
            Should be at least 1.
        data_table (astropy.table.Table): output from
            ``orbitize.read_input.read_file()``
        stellar_or_system_mass (float): mass of the primary star (if fitting for
            dynamical masses of both components, for example when you have both 
            astrometry and RVs) or total system mass (if fitting for total system 
            mass only, as in the case of a vanilla 2-body fit using relative 
            astrometry only ) [M_sol]
        plx (float): mean parallax of the system, in mas
        mass_err (float, optional): uncertainty on ``stellar_or_system_mass``, in M_sol
        plx_err (float, optional): uncertainty on ``plx``, in mas
        restrict_angle_ranges (bool, optional): if True, restrict the ranges
            of the position angle of nodes to [0,180)
            to get rid of symmetric double-peaks for imaging-only datasets.
        tau_ref_epoch (float, optional): reference epoch for defining tau (MJD).
            Default is 58849 (Jan 1, 2020).
        fit_secondary_mass (bool, optional): if True, include the dynamical
            mass of the orbiting body as a fitted parameter. If this is set to
            False, ``stellar_or_system_mass`` is taken to be the total mass of the system.
            (default: False)
        hipparcos_IAD (orbitize.hipparcos.HipparcosLogProb): an object
            containing information & precomputed values relevant to Hipparcos
            IAD fitting. See hipparcos.py for more details.
        gaia (orbitize.gaia.GaiaLogProb): an object
            containing information & precomputed values relevant to Gaia
            astrometrry fitting. See gaia.py for more details.
        fitting_basis (str): the name of the class corresponding to the fitting
            basis to be used. See basis.py for a list of implemented fitting bases.
        use_rebound (bool): if True, use an n-body backend solver instead
            of a Keplerian solver.

    Priors are initialized as a list of orbitize.priors.Prior objects and stored
    in the variable ``System.sys_priors``. You should initialize this class,
    then overwrite priors you wish to customize. You can use the
    ``System.param_idx`` attribute to figure out which indices correspond to
    which fitting parameters. See the "changing priors" tutorial for more detail.

    Written: Sarah Blunt, Henry Ngo, Jason Wang, 2018
    """

    def __init__(
        self,
        num_secondary_bodies,
        data_table,
        stellar_or_system_mass,
        plx,
        mass_err=0,
        plx_err=0,
        restrict_angle_ranges=False,
        tau_ref_epoch=58849,
        fit_secondary_mass=False,
        hipparcos_IAD=None,
        gaia=None,
        fitting_basis="Standard",
        use_rebound=False,
        light_timing_correction=False,
    ):
        self.num_secondary_bodies = num_secondary_bodies
        self.data_table = data_table
        self.stellar_or_system_mass = stellar_or_system_mass
        self.plx = plx
        self.mass_err = mass_err
        self.plx_err = plx_err
        self.restrict_angle_ranges = restrict_angle_ranges
        self.tau_ref_epoch = tau_ref_epoch
        self.fit_secondary_mass = fit_secondary_mass
        self.hipparcos_IAD = hipparcos_IAD
        self.gaia = gaia
        self.fitting_basis = fitting_basis
        self.use_rebound = use_rebound
        self.full_perspective_model = fitting_basis == "FullPerspectiveModel"
        self.light_timing_correction = light_timing_correction

        if self.light_timing_correction and not self.full_perspective_model:
            raise ValueError("Corrections for light timing variations can only be used with full perspective and "
                             "proper motion modeling")

        self.best_epochs = []
        self.input_table = self.data_table.copy()

        # Group the data in some useful ways

        self.body_indices = []

        # List of arrays of indices corresponding to epochs in RA/Dec for each body
        self.radec = []

        # List of arrays of indices corresponding to epochs in SEP/PA for each body
        self.seppa = []

        # List of index arrays corresponding to each rv for each body
        self.rv = []

        self.fit_astrometry = True
        radec_indices = np.where(self.data_table["quant_type"] == "radec")
        seppa_indices = np.where(self.data_table["quant_type"] == "seppa")

        if len(radec_indices[0]) == 0 and len(seppa_indices[0]) == 0:
            self.fit_astrometry = False
        rv_indices = np.where(self.data_table["quant_type"] == "rv")

        # defining all indices to loop through the unique rv instruments to get different offsets and jitters
        instrument_list = np.unique(self.data_table["instrument"])
        inst_indices_all = []
        for inst in instrument_list:
            inst_indices = np.where(self.data_table["instrument"] == inst)
            inst_indices_all.append(inst_indices)

        # defining indices for unique instruments in the data table
        self.rv_instruments = np.unique(self.data_table["instrument"][rv_indices])
        self.rv_inst_indices = []
        for inst in self.rv_instruments:
            inst_indices = np.where(self.data_table["instrument"] == inst)
            self.rv_inst_indices.append(inst_indices)

        # astrometry instruments same for radec and seppa:
        self.astr_instruments = np.unique(
            self.data_table["instrument"][
                np.where(self.data_table["quant_type"] != "rv")
            ]
        )
        # save indicies for all of the ra/dec, sep/pa measurements for convenience
        self.all_radec = radec_indices
        self.all_seppa = seppa_indices

        for body_num in np.arange(self.num_secondary_bodies + 1):
            self.body_indices.append(np.where(self.data_table["object"] == body_num))

            self.radec.append(
                np.intersect1d(self.body_indices[body_num], radec_indices)
            )
            self.seppa.append(
                np.intersect1d(self.body_indices[body_num], seppa_indices)
            )
            self.rv.append(np.intersect1d(self.body_indices[body_num], rv_indices))

        # we should track the influence of the planet(s) on each other/the star if:
        # we are not fitting massless planets and
        # we have more than 1 companion OR we have stellar astrometry
        self.track_planet_perturbs = self.fit_secondary_mass and (
            (
                ((len(self.radec[0]) + len(self.seppa[0])) > 0)
                or (self.num_secondary_bodies > 1)
                or (hipparcos_IAD is not None)
                or (gaia is not None)
            )
        )

        if self.hipparcos_IAD is not None:
            self.track_planet_perturbs = True

        if self.restrict_angle_ranges:
            angle_upperlim = np.pi
        else:
            angle_upperlim = 2.0 * np.pi

        # Check for rv data
        contains_rv = False
        if len(self.rv[0]) > 0:
            contains_rv = True

        # Assign priors for the given basis set
        self.extra_basis_kwargs = {}
        basis_obj = getattr(basis, self.fitting_basis)

        # Obtain extra necessary data to assign priors for XYZ
        if self.fitting_basis == "XYZ":
            # Get epochs with least uncertainty, as is done in sampler.py
            convert_warning_print = False
            for body_num in np.arange(self.num_secondary_bodies) + 1:
                if len(self.radec[body_num]) > 0:
                    # only print the warning once.
                    if not convert_warning_print:
                        print(
                            "Converting ra/dec data points in data_table to sep/pa. Original data are stored in input_table."
                        )
                        convert_warning_print = True
                    self.convert_data_table_radec2seppa(body_num=body_num)

            sep_err = self.data_table[
                np.where(self.data_table["quant_type"] == "seppa")
            ]["quant1_err"].copy()
            meas_object = self.data_table[
                np.where(self.data_table["quant_type"] == "seppa")
            ]["object"].copy()

            astr_inds = np.where(self.input_table["object"] > 0)[0]
            astr_data = self.input_table[astr_inds]
            epochs = astr_data["epoch"]

            self.best_epochs = []
            self.best_epoch_idx = []
            min_sep_indices = np.argsort(
                sep_err
            )  # indices of sep err sorted from smallest to higheset
            min_sep_indices_body = meas_object[
                min_sep_indices
            ]  # the corresponding body_num that these sorted measurements correspond to
            for i in range(self.num_secondary_bodies):
                body_num = i + 1
                this_object_meas = np.where(min_sep_indices_body == body_num)[0]
                if np.size(this_object_meas) == 0:
                    # no data, no scaling
                    self.best_epochs.append(None)
                    continue
                # get the smallest measurement belonging to this body
                this_best_epoch_idx = min_sep_indices[this_object_meas][
                    0
                ]  # already sorted by argsort
                self.best_epoch_idx.append(this_best_epoch_idx)
                this_best_epoch = epochs[this_best_epoch_idx]
                self.best_epochs.append(this_best_epoch)

            self.extra_basis_kwargs = {
                "data_table": astr_data,
                "best_epoch_idx": self.best_epoch_idx,
                "epochs": epochs,
            }

        self.basis = basis_obj(
            self.stellar_or_system_mass,
            self.mass_err,
            self.plx,
            self.plx_err,
            self.num_secondary_bodies,
            self.fit_secondary_mass,
            angle_upperlim=angle_upperlim,
            hipparcos_IAD=self.hipparcos_IAD,
            rv=contains_rv,
            rv_instruments=self.rv_instruments,
            **self.extra_basis_kwargs
        )

        self.basis.verify_params()
        self.sys_priors, self.labels = self.basis.construct_priors()

        # if we're fitting absolute astrometry of the star, create an object that
        # knows how to compute predicted astrometric motion due to parallax and
        # proper motion

        if (len(self.radec[0]) + len(self.seppa[0])) > 0:
            self.stellar_astrom_epochs = self.data_table["epoch"][
                (self.data_table["quant_type"] == "radec")
                & (self.data_table["object"] == 0)
            ]
            alpha0 = self.hipparcos_IAD.alpha0
            delta0 = self.hipparcos_IAD.delta0
            alphadec0_epoch = self.hipparcos_IAD.alphadec0_epoch
            self.pm_plx_predictor = hipparcos.PMPlx_Motion(
                self.stellar_astrom_epochs, alpha0, delta0, alphadec0_epoch
            )

        self.secondary_mass_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("m") and not i.endswith("0"))
        ]

        self.sma_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("sma"))
        ]
        self.ecc_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("ecc"))
        ]
        self.inc_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("inc"))
        ]
        self.aop_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("aop"))
        ]
        self.pan_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("pan"))
        ]
        self.tau_indx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("tau"))
        ]
        self.mpl_idx = [
            self.basis.standard_basis_idx[i]
            for i in self.basis.standard_basis_idx.keys()
            if (i.startswith("m") and i[1:] not in ["tot", "0"])
        ]

        self.param_idx = self.basis.param_idx

    def save(self, hf):
        """
        Saves the current object to an hdf5 file

        Args:
            hf (h5py._hl.files.File): a currently open hdf5 file in which
                to save the object.
        """

        hf.attrs["num_secondary_bodies"] = self.num_secondary_bodies

        hf.create_dataset("data", data=self.input_table)

        hf.attrs["restrict_angle_ranges"] = self.restrict_angle_ranges
        hf.attrs["tau_ref_epoch"] = self.tau_ref_epoch
        hf.attrs["stellar_or_system_mass"] = self.stellar_or_system_mass
        hf.attrs["plx"] = self.plx
        hf.attrs["mass_err"] = self.mass_err
        hf.attrs["plx_err"] = self.plx_err
        hf.attrs["fit_secondary_mass"] = self.fit_secondary_mass

        if self.gaia is not None:
            self.gaia._save(hf)
        elif self.hipparcos_IAD is not None:
            self.hipparcos_IAD._save(hf)
        hf.attrs["fitting_basis"] = self.fitting_basis
        hf.attrs["use_rebound"] = self.use_rebound

    def compute_all_orbits(self, params_arr, epochs=None, comp_rebound=False):
        """
        Calls orbitize.kepler.calc_orbit and optionally accounts for multi-body
        interactions. Also computes total quantities like RV (without jitter/gamma)

        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.
            epochs (np.array of float): epochs (in mjd) at which to compute
                orbit predictions.
            comp_rebound (bool, optional): A secondary optional input for
                use of N-body solver Rebound; by default, this will be set
                to false and a Kepler solver will be used instead.

        Returns:
            tuple:

                raoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    RA offsets from barycenter at each epoch.

                decoff (np.array of float): N_epochs x N_bodies x N_orbits array of
                    Dec offsets from barycenter at each epoch.

                vz (np.array of float): N_epochs x N_bodies x N_orbits array of
                    radial velocities at each epoch.

        """

        if epochs is None:
            epochs = self.data_table["epoch"]

        n_epochs = len(epochs)

        if len(params_arr.shape) == 1:
            n_orbits = 1
        else:
            n_orbits = params_arr.shape[1]

        ra_kepler = np.zeros(
            (n_epochs, self.num_secondary_bodies + 1, n_orbits)
        )  # N_epochs x N_bodies x N_orbits
        dec_kepler = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        ra_perturb = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))
        dec_perturb = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        vz = np.zeros((n_epochs, self.num_secondary_bodies + 1, n_orbits))

        # mass/mtot used to compute each Keplerian orbit will be needed later to compute perturbations
        if self.track_planet_perturbs:
            masses = np.zeros((self.num_secondary_bodies + 1, n_orbits))
            mtots = np.zeros((self.num_secondary_bodies + 1, n_orbits))

        if comp_rebound or self.use_rebound:
            sma = params_arr[self.sma_indx]
            ecc = params_arr[self.ecc_indx]
            inc = params_arr[self.inc_indx]
            argp = params_arr[self.aop_indx]
            lan = params_arr[self.pan_indx]
            tau = params_arr[self.tau_indx]
            plx = params_arr[self.basis.standard_basis_idx["plx"]]

            if self.fit_secondary_mass:
                m_pl = params_arr[self.mpl_idx]
                m0 = params_arr[self.basis.param_idx["m0"]]
                mtot = m0 + sum(m_pl)
            else:
                m_pl = np.zeros(self.num_secondary_bodies)
                # if not fitting for secondary mass, then total mass must be stellar mass
                mtot = params_arr[self.basis.param_idx["mtot"]]

            raoff, deoff, vz = nbody.calc_orbit(
                epochs,
                sma,
                ecc,
                inc,
                argp,
                lan,
                tau,
                plx,
                mtot,
                tau_ref_epoch=self.tau_ref_epoch,
                m_pl=m_pl,
                output_star=True,
            )

        else:
            for body_num in np.arange(self.num_secondary_bodies) + 1:
                sma = params_arr[
                    self.basis.standard_basis_idx["sma{}".format(body_num)]
                ]
                ecc = params_arr[
                    self.basis.standard_basis_idx["ecc{}".format(body_num)]
                ]
                inc = params_arr[
                    self.basis.standard_basis_idx["inc{}".format(body_num)]
                ]
                argp = params_arr[
                    self.basis.standard_basis_idx["aop{}".format(body_num)]
                ]
                lan = params_arr[
                    self.basis.standard_basis_idx["pan{}".format(body_num)]
                ]
                tau = params_arr[
                    self.basis.standard_basis_idx["tau{}".format(body_num)]
                ]
                plx = params_arr[self.basis.standard_basis_idx["plx"]]

                if self.fit_secondary_mass:
                    # mass of secondary bodies are in order from -1-num_bodies until -2 in order.
                    mass = params_arr[
                        self.basis.standard_basis_idx["m{}".format(body_num)]
                    ]
                    m0 = params_arr[self.basis.standard_basis_idx["m0"]]

                    # For what mtot to use to calculate central potential, we should use the mass enclosed in a sphere with r <= distance of planet.
                    # We need to select all planets with sma < this planet.
                    all_smas = params_arr[self.sma_indx]
                    within_orbit = np.where(all_smas <= sma)
                    outside_orbit = np.where(all_smas > sma)
                    all_pl_masses = params_arr[self.secondary_mass_indx]
                    inside_masses = all_pl_masses[within_orbit]
                    mtot = np.sum(inside_masses) + m0

                else:
                    m_pl = np.zeros(self.num_secondary_bodies)
                    # if not fitting for secondary mass, then total mass must be stellar mass
                    mass = None
                    m0 = None
                    mtot = params_arr[self.basis.standard_basis_idx["mtot"]]

                if self.track_planet_perturbs:
                    masses[body_num] = mass
                    mtots[body_num] = mtot

                # solve Kepler's equation
                if not self.full_perspective_model:
                    raoff, decoff, vz_i = kepler.calc_orbit(
                        epochs,
                        sma,
                        ecc,
                        inc,
                        argp,
                        lan,
                        tau,
                        plx,
                        mtot,
                        mass_for_Kamp=m0,
                        tau_ref_epoch=self.tau_ref_epoch,
                    )
                else:
                    if self.hipparcos_IAD is not None:
                        ra_com_i = (params_arr[self.basis.standard_basis_idx['alpha0']] * u.mas).to(u.deg) / np.cos(
                            np.radians(self.hipparcos_IAD.delta0)) + self.hipparcos_IAD.alpha0
                        dec_com_i = (params_arr[self.basis.standard_basis_idx['delta0']] * u.mas).to(
                            u.deg) + self.hipparcos_IAD.delta0
                    else:
                        ra_com_i = params_arr[self.basis.standard_basis_idx['alpha0']]
                        dec_com_i = params_arr[self.basis.standard_basis_idx['delta0']]
                    print(ra_com_i)
                    pmra_com_i = params_arr[self.basis.standard_basis_idx['pmra']]
                    pmdec_com_i = params_arr[self.basis.standard_basis_idx['pmdec']]
                    plx_com_i = params_arr[self.basis.standard_basis_idx['plx']]
                    rv_com_i = params_arr[self.basis.standard_basis_idx['ref_rv']]
                    ref_epoch = self.basis.perspective_ref_epoch
                    print(ref_epoch)
                    if self.light_timing_correction:
                        use_epochs = light_timing_epoch_correction(epochs, ra_com_i, dec_com_i, plx_com_i, pmra_com_i,
                                                                   pmdec_com_i, rv_com_i,
                                                                   ref_epoch)
                    else:
                        use_epochs = deepcopy(epochs)

                    raoff, decoff, vz_i = calc_corrected_orbit(sma, ecc, inc, argp, lan, tau, plx, mtot, m0,
                                                               self.tau_ref_epoch, use_epochs, ref_epoch, ra_com_i,
                                                               dec_com_i, plx_com_i, pmra_com_i, pmdec_com_i,
                                                               rv_com_i, mass, n_orbits)

                # raoff, decoff, vz are scalers if the length of epochs is 1
                if len(epochs) == 1:
                    raoff = np.array([raoff])
                    decoff = np.array([decoff])
                    vz_i = np.array([vz_i])

                # add Keplerian ra/deoff for this body to storage arrays
                ra_kepler[:, body_num, :] = np.reshape(raoff, (n_epochs, n_orbits))
                dec_kepler[:, body_num, :] = np.reshape(decoff, (n_epochs, n_orbits))
                vz[:, body_num, :] = np.reshape(vz_i, (n_epochs, n_orbits))

                # vz_i is the ith companion radial velocity
                if self.fit_secondary_mass:
                    vz0 = np.reshape(
                        vz_i * -(mass / m0), (n_epochs, n_orbits)
                    )  # calculating stellar velocity due to ith companion
                    vz[:, 0, :] += vz0  # adding stellar velocity and gamma

            # if we are fitting for the mass of the planets, then they will perturb the star
            # add the perturbation on the star due to this planet on the relative astrometry of the planet that was measured
            # We are superimposing the Keplerian orbits, so we can add it linearly, scaled by the mass.
            # Because we are in Jacobi coordinates, for companions, we only should model the effect of planets interior to it.
            # (Jacobi coordinates mean that separation for a given companion is measured relative to the barycenter of all interior companions)
            if self.track_planet_perturbs and not self.full_perspective_model:
                for body_num in np.arange(self.num_secondary_bodies + 1):
                    if body_num > 0:
                        # for companions, only perturb companion orbits at larger SMAs than this one.
                        sma = params_arr[
                            self.basis.standard_basis_idx["sma{}".format(body_num)]
                        ]
                        all_smas = params_arr[self.sma_indx]
                        outside_orbit = np.where(all_smas > sma)[0]
                        which_perturb_bodies = outside_orbit + 1

                        # the planet will also perturb the star
                        which_perturb_bodies = np.append([0], which_perturb_bodies)

                    else:
                        # for the star, what we are measuring is its position relative to the system barycenter
                        # so we want to account for all of the bodies.
                        which_perturb_bodies = np.arange(self.num_secondary_bodies + 1)

                    for other_body_num in which_perturb_bodies:
                        # skip itself since the the 2-body problem is measuring the planet-star separation already
                        if (body_num == other_body_num) | (body_num == 0):
                            continue

                        ## NOTE: we are only handling astrometry right now (TODO: integrate RV into this)
                        # this computes the perturbation on the other body due to the current body

                        # star is perturbed in opposite direction
                        if other_body_num == 0:
                            ra_perturb[:, other_body_num, :] -= (
                                masses[body_num] / mtots[body_num]
                            ) * ra_kepler[:, body_num, :]
                            dec_perturb[:, other_body_num, :] -= (
                                masses[body_num] / mtots[body_num]
                            ) * dec_kepler[:, body_num, :]

                        else:
                            ra_perturb[:, other_body_num, :] += (
                                masses[body_num] / mtots[body_num]
                            ) * ra_kepler[:, body_num, :]
                            dec_perturb[:, other_body_num, :] += (
                                masses[body_num] / mtots[body_num]
                            ) * dec_kepler[:, body_num, :]

            raoff = ra_kepler + ra_perturb
            deoff = dec_kepler + dec_perturb

        if self.fitting_basis == "XYZ":
            # Find and filter out unbound orbits
            bad_orbits = np.where(np.logical_or(ecc >= 1.0, ecc < 0.0))[0]
            if bad_orbits.size != 0:
                raoff[:, :, bad_orbits] = np.inf
                deoff[:, :, bad_orbits] = np.inf
                vz[:, :, bad_orbits] = np.inf
                return raoff, deoff, vz
            else:
                return raoff, deoff, vz
        else:
            return raoff, deoff, vz

    def compute_model(self, params_arr, use_rebound=False):
        """
        Compute model predictions for an array of fitting parameters.
        Calls the above compute_all_orbits() function, adds jitter/gamma to
        RV measurements, and propagates these predictions to a model array that
        can be subtracted from a data array to compute chi2.

        Args:
            params_arr (np.array of float): RxM array
                of fitting parameters, where R is the number of
                parameters being fit, and M is the number of orbits
                we need model predictions for. Must be in the same order
                documented in ``System()`` above. If M=1, this can be a 1d array.
            use_rebound (bool, optional): A secondary optional input for
                use of N-body solver Rebound; by default, this will be set
                to false and a Kepler solver will be used instead.

        Returns:
            tuple of:
                np.array of float: Nobsx2xM array model predictions. If M=1, this is
                    a 2d array, otherwise it is a 3d array.
                np.array of float: Nobsx2xM array jitter predictions. If M=1, this is
                    a 2d array, otherwise it is a 3d array.
        """

        to_convert = np.copy(params_arr)
        standard_params_arr = self.basis.to_standard_basis(to_convert)

        if use_rebound:
            raoff, decoff, vz = self.compute_all_orbits(
                standard_params_arr, comp_rebound=True
            )
        else:
            raoff, decoff, vz = self.compute_all_orbits(standard_params_arr)

        if len(standard_params_arr.shape) == 1:
            n_orbits = 1
        else:
            n_orbits = standard_params_arr.shape[1]

        n_epochs = len(self.data_table)
        model = np.zeros((n_epochs, 2, n_orbits))
        jitter = np.zeros((n_epochs, 2, n_orbits))
        gamma = np.zeros((n_epochs, 2, n_orbits))

        if len(self.rv[0]) > 0 and self.fit_secondary_mass:
            # looping through instruments to get the gammas & jitters
            for rv_idx in range(len(self.rv_instruments)):
                jitter[self.rv_inst_indices[rv_idx], 0] = standard_params_arr[  # [km/s]
                    self.basis.standard_basis_idx[
                        "sigma_{}".format(self.rv_instruments[rv_idx])
                    ]
                ]
                jitter[self.rv_inst_indices[rv_idx], 1] = np.nan

                gamma[self.rv_inst_indices[rv_idx], 0] = standard_params_arr[
                    self.basis.standard_basis_idx[
                        "gamma_{}".format(self.rv_instruments[rv_idx])
                    ]
                ]
                gamma[self.rv_inst_indices[rv_idx], 1] = np.nan

        for body_num in np.arange(self.num_secondary_bodies + 1):
            # for the model points that correspond to this planet's orbit, add the model prediction
            # RA/Dec
            if len(self.radec[body_num]) > 0:  # (prevent empty array dimension errors)
                model[self.radec[body_num], 0] = raoff[
                    self.radec[body_num], body_num, :
                ]  # N_epochs x N_bodies x N_orbits
                model[self.radec[body_num], 1] = decoff[
                    self.radec[body_num], body_num, :
                ]

            # Sep/PA
            if len(self.seppa[body_num]) > 0:
                sep, pa = radec2seppa(raoff, decoff)

                model[self.seppa[body_num], 0] = sep[self.seppa[body_num], body_num, :]
                model[self.seppa[body_num], 1] = pa[self.seppa[body_num], body_num, :]

            # RV
            if len(self.rv[body_num]) > 0:
                model[self.rv[body_num], 0] = vz[self.rv[body_num], body_num, :]
                model[self.rv[body_num], 1] = np.nan

        # if we have abs astrometry measurements in the input file (i.e. not
        # from Hipparcos or Gaia), add the parallactic & proper motion here by
        # calling AbsAstrom compute_model
        if len(self.radec[0]) > 0:
            ra_pred, dec_pred = self.pm_plx_predictor.compute_astrometric_model(
                params_arr, self.param_idx
            )

            # divide by cos(delta0) because orbitize! input is delta(ra), not
            # delta(ra)*cos(delta0)
            model[self.radec[0], 0] += ra_pred.reshape(model[self.radec[0], 0].shape) / np.cos(np.radians(self.pm_plx_predictor.delta0))
            model[self.radec[0], 1] += dec_pred.reshape(model[self.radec[0], 0].shape)

        if n_orbits == 1:
            model = model.reshape((n_epochs, 2))
            jitter = jitter.reshape((n_epochs, 2))
            gamma = gamma.reshape((n_epochs, 2))

        if self.fit_secondary_mass:
            return model + gamma, jitter
        else:
            return model, jitter

    def convert_data_table_radec2seppa(self, body_num=1):
        """
        Converts rows of self.data_table given in radec to seppa.
        Note that self.input_table remains unchanged.

        Args:
            body_num (int): which object to convert (1 = first planet)
        """
        for i in self.radec[
            body_num
        ]:  # Loop through rows where input provided in radec
            # Get ra/dec values
            ra = self.data_table["quant1"][i]
            ra_err = self.data_table["quant1_err"][i]
            dec = self.data_table["quant2"][i]
            dec_err = self.data_table["quant2_err"][i]
            radec_corr = self.data_table["quant12_corr"][i]
            # Convert to sep/PA
            sep, pa = radec2seppa(ra, dec)

            if np.isnan(radec_corr):
                # E-Z
                sep_err = 0.5 * (ra_err + dec_err)
                pa_err = np.degrees(sep_err / sep)
                seppa_corr = np.nan
            else:
                sep_err, pa_err, seppa_corr = transform_errors(
                    ra, dec, ra_err, dec_err, radec_corr, radec2seppa
                )

            # Update data_table
            self.data_table["quant1"][i] = sep
            self.data_table["quant1_err"][i] = sep_err
            self.data_table["quant2"][i] = pa
            self.data_table["quant2_err"][i] = pa_err
            self.data_table["quant12_corr"][i] = seppa_corr
            self.data_table["quant_type"][i] = "seppa"
            # Update self.radec and self.seppa arrays
            self.radec[body_num] = np.delete(
                self.radec[body_num], np.where(self.radec[body_num] == i)[0]
            )
            self.seppa[body_num] = np.append(self.seppa[body_num], i)


def radec2seppa(ra, dec, mod180=False):
    """
    Convenience function for converting from
    right ascension/declination to separation/
    position angle.

    Args:
        ra (np.array of float): array of RA values, in mas
        dec (np.array of float): array of Dec values, in mas
        mod180 (Bool): if True, output PA values will be given
            in range [180, 540) (useful for plotting short
            arcs with PAs that cross 360 during observations)
            (default: False)


    Returns:
        tuple of float: (separation [mas], position angle [deg])

    """
    sep = np.sqrt((ra**2) + (dec**2))
    pa = np.degrees(np.arctan2(ra, dec)) % 360.0

    if mod180:
        pa[pa < 180] += 360

    return sep, pa


def seppa2radec(sep, pa):
    """
    Convenience function to convert sep/pa to ra/dec

    Args:
        sep (np.array of float): array of separation in mas
        pa (np.array of float): array of position angles in degrees

    Returns:
        tuple: (ra [mas], dec [mas])
    """
    ra = sep * np.sin(np.radians(pa))
    dec = sep * np.cos(np.radians(pa))

    return ra, dec


def transform_errors(x1, x2, x1_err, x2_err, x12_corr, transform_func, nsamps=100000):
    """
     Transform errors and covariances from one basis to another using a Monte Carlo
     apporach

    Args:
         x1 (float): planet location in first coordinate (e.g., RA, sep) before
             transformation
         x2 (float): planet location in the second coordinate (e.g., Dec, PA)
             before transformation)
         x1_err (float): error in x1
         x2_err (float): error in x2
         x12_corr (float): correlation between x1 and x2
         transform_func (function): function that transforms between (x1, x2)
             and (x1p, x2p) (the transformed coordinates). The function signature
             should look like: `x1p, x2p = transform_func(x1, x2)`
         nsamps (int): number of samples to draw more the Monte Carlo approach.
             More is slower but more accurate.
     Returns:
         tuple (x1p_err, x2p_err, x12p_corr): the errors and correlations for
             x1p,x2p (the transformed coordinates)
    """

    if np.isnan(x12_corr):
        x12_corr = 0.0

    # construct covariance matrix from the terms provided
    cov = np.array(
        [
            [x1_err**2, x1_err * x2_err * x12_corr],
            [x1_err * x2_err * x12_corr, x2_err**2],
        ]
    )

    samps = np.random.multivariate_normal([x1, x2], cov, size=nsamps)

    x1p, x2p = transform_func(samps[:, 0], samps[:, 1])

    x1p_err = np.std(x1p)
    x2p_err = np.std(x2p)
    x12_corr = np.corrcoef([x1p, x2p])[0, 1]

    return x1p_err, x2p_err, x12_corr


def generate_synthetic_data(
    orbit_frac,
    mtot,
    plx,
    ecc=0.5,
    inc=np.pi / 4,
    argp=np.pi / 4,
    lan=np.pi / 4,
    tau=0.8,
    num_obs=4,
    unc=2,
):
    """Generate an orbitize-table of synethic data

    Args:
        orbit_frac (float): percentage of orbit covered by synthetic data
        mtot (float): total mass of the system [M_sol]
        plx (float): parallax of system [mas]
        num_obs (int): number of observations to generate
        unc (float): uncertainty on all simulated RA & Dec measurements [mas]

    Returns:
        2-tuple:
            - `astropy.table.Table`: data table of generated synthetic data
            - float: the semimajor axis of the generated data
    """

    # calculate RA/Dec at three observation epochs
    # `num_obs` epochs between ~2000 and ~2003 [MJD]
    observation_epochs = np.linspace(51550.0, 52650.0, num_obs)
    num_obs = len(observation_epochs)

    # calculate the orbital fraction
    orbit_coverage = (max(observation_epochs) - min(observation_epochs)) / 365.25
    period = 100 * orbit_coverage / orbit_frac
    sma = (period**2 * mtot) ** (1 / 3)

    # calculate RA/Dec at three observation epochs
    # `num_obs` epochs between ~2000 and ~2003 [MJD]
    ra, dec, _ = kepler.calc_orbit(
        observation_epochs, sma, ecc, inc, argp, lan, tau, plx, mtot
    )

    # add Gaussian noise to simulate measurement
    ra += np.random.normal(scale=unc, size=num_obs)
    dec += np.random.normal(scale=unc, size=num_obs)

    # define observational uncertainties
    ra_err = dec_err = np.ones(num_obs) * unc

    data_table = table.Table(
        [observation_epochs, [1] * num_obs, ra, ra_err, dec, dec_err],
        names=("epoch", "object", "raoff", "raoff_err", "decoff", "decoff_err"),
    )
    # read into orbitize format
    data_table = read_file(data_table)

    return data_table, sma


def radecz2xyz(ra, dec, parallax, pmra, pmdec, rv):
    """
    Converts from ra, dec, z coordinates to x, y, z coordinates. x points in the direction where
    ra = dec = 0, z is perpendicular to the celestial equator, y is such that it forms a right-handed
    coordinate system. These values should be instanteneous.
    Args:
        ra: RA of the location to convert (deg)
        dec: Dec of the location to convert (deg)
        parallax: parallax of the location to convert (mas)
        pmra: proper motion in the ra direction (mas/yr)
        pmdec: proper motion in the dec direction (mas/yr)
        rv: radial velocity (km/s)
    Returns:

    """
    my206265 = 180 / np.pi * 60 * 60
    sec2year = (1*u.yr).to(u.s).value
    pc2km = (1*u.pc).to(u.km).value

    distance = 1 / np.tan(parallax / 1000 / my206265) / my206265

    # convert RV to pc/year, convert delta RA and delta Dec to radians/year

    dra = pmra / 1000 / my206265 / np.cos(dec * np.pi / 180)
    ddec = pmdec / 1000 / my206265
    ddist = rv / pc2km * sec2year

    # convert first epoch to x,y,z and dx,dy,dz

    x = np.cos(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * distance
    y = np.sin(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * distance
    z = np.sin(dec * np.pi / 180) * distance

    # Excellent.  Now dx,dy,dz,which are constants

    dx = -1 * np.sin(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * distance * dra - np.cos(ra * np.pi / 180) * np.sin(
        dec * np.pi / 180) * distance * ddec + np.cos(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * ddist

    dy = 1 * np.cos(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * distance * dra - np.sin(ra * np.pi / 180) * np.sin(
        dec * np.pi / 180) * distance * ddec + np.sin(ra * np.pi / 180) * np.cos(dec * np.pi / 180) * ddist

    dz = 1 * np.cos(dec * np.pi / 180) * distance * ddec + np.sin(dec * np.pi / 180) * ddist

    return x, y, z, dx, dy, dz


def xyz2radecz(x, y, z, dx, dy, dz):
    """
    Converts from x, y, z coordinate system to ra, dec, z coordinate system. Convention for
    cartesian system is described in the function above (radecz2xyz).
    Args:
        x: x coordinate to convert (parsec)
        y: y coordinate to convert (parsec)
        z: z coordinate to convert (parsec)
        dx: dx coordinate to convert (parsec/yr)
        dy: dy coordinate to convert (parsec/yr)
        dz: dz coordinate to convert (parsec/yr)
    Returns:

    """
    my206265 = 180 / np.pi * 60 * 60
    sec2year = (1*u.yr).to(u.s).value
    pc2km = (1*u.pc).to(u.km).value

    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    parallax = np.arctan(1 / my206265 / distance) * my206265 * 1000

    ra = ((np.arctan2(y, x) / (np.pi / 180) + 360) % 360)
    dec = np.arcsin(z / distance) / (np.pi / 180)

    ddist = 1 / np.sqrt(x ** 2 + y ** 2 + z ** 2) * (x * dx + y * dy + z * dz)
    dra = 1 / (x ** 2 + y ** 2) * (-1 * y * dx + x * dy)
    ddec = 1 / (distance * np.sqrt(1 - z ** 2 / distance ** 2)) * (-1 * z * ddist / distance + dz)

    pmra = dra * my206265 * 1000 * np.cos(dec * (np.pi / 180))
    pmdec = ddec * 1000 * my206265
    rv = ddist * pc2km / sec2year

    return ra, dec, parallax, pmra, pmdec, rv


def rotate(ra, dec, xp, yp, zp):
    """
    Rotates between the celestial cartesian plane (x points where ra=dec=0, z perpendicular to celestial
    equator, y made to form right-handed coordinate system) and the orbitize cartesian plane (z points
    directly toward star system, x points in local RA direction, y points in local Dec direction).
    Args:
        ra: RA at the reference epoch (i.e. the epoch when the parameters orbitize is evaluating are true)
        dec: Dec at the reference epoch
        xp: x value in the primed (orbitize) coordinate system
        yp: y value in the primed (orbitize) coordinate system
        zp: z value in the primed (orbitize) coordinate system
    Returns:
        x: x value in the unprimed (celestial) coordinate frame
        y: y value in the unprimed (celestial) coordinate frame
        z: z value in the unprimed (celestial) coordinate frame

    """
    x = -np.sin(ra * np.pi / 180) * xp - np.cos(ra * np.pi / 180) * np.sin(dec * np.pi / 180) * yp + np.cos(
        ra * np.pi / 180) * np.cos(dec * np.pi / 180) * zp
    y = np.cos(ra * np.pi / 180) * xp - np.sin(ra * np.pi / 180) * np.sin(dec * np.pi / 180) * yp + np.sin(
        ra * np.pi / 180) * np.cos(dec * np.pi / 180) * zp
    z = np.cos(dec * np.pi / 180) * yp + np.sin(dec * np.pi / 180) * zp
    return x, y, z


def transform_motion(epochs, ra_i, dec_i, parallax_i, sys_pmra, sys_pmdec, sys_rv,
                     orbit_xdotp, orbit_ydotp, orbit_zdotp, orbit_xp, orbit_yp, orbit_zp,
                     ra_com_i, dec_com_i, ref_epoch,
                     orbit_xp_ref, orbit_yp_ref, orbit_zp_ref):
    """
    Transforms stellar motion, taking into account changing distance and perspective. Rigorous
    transformation of coordinates for stars with significant proper motion.

    Args:
        epochs (array): epochs (in MJD) at which to calculate the rigorous transformation of coordinates
        ra_i (float): initial RA (deg, at ref_epoch) of the body to transform
        dec_i (float): initial Dec (deg, at ref_epoch) of the body to transform
        parallax_i (float): initial parallax (mas, at ref_epoch) of the body to transform
        sys_pmra (float): proper motion in the RA direction (mas/yr) at the ref_epoch for the entire system
        sys_pmdec (float): proper motion in the Dec direction (mas/yr) at the ref_epoch for the entire system
        sys_rv (float): RV of the system at ref_epoch (km/s)
        orbit_xp (array): orbital motion in the primed (orbitize output) x direction
        orbit_yp (array): orbital motion in the primed (orbitize output) y direction
        orbit_zp (array): orbital motion in the primed (orbitize output) z direction
        ra_com_i (float): RA (deg) of the center of mass (where orbitize assumes it is looking) at ref_epoch
        dec_com_i (float): Dec (deg) of the center of mass (where orbitize assumes it is looking) at ref_epoch
        ref_epoch (float): epoch (MJD) on which many other values must be true (including orbital parameters)
        orbit_xp_ref (float): x coordinate of the orbital motion in the primed frame on ref_epoch
        orbit_yp_ref (float): y coordinate of the orbital motion in the primed frame on ref_epoch
        orbit_zp_ref (float): z coordinate of the orbital motion in the primed frame on ref_epoch
    Returns:
        ra_f (array): final (transformed) RA values at each epoch in epochs (deg)
        dec_f (array): final (transformed) Dec values at each epoch in epochs (deg)
        parallax_f (array): final (transformed) parallax values at each epoch in epochs (mas)
        pmra_f (array): final (transformed) RA proper motion at each epoch in epochs (mas/yr)
        pmdec_f (array): final (transformed) Dec proper motion at each epoch in epochs (mas/yr)
        rv_f (array): final (transformed) RV values at each epoch in epochs (km/s)
    """
    del_epochs = ((epochs - ref_epoch) * u.day).to(u.yr).value
    x_i, y_i, z_i, dx_sys, dy_sys, dz_sys = radecz2xyz(ra_i, dec_i, parallax_i, sys_pmra, sys_pmdec, sys_rv)
    orbit_x, orbit_y, orbit_z = rotate(ra_com_i, dec_com_i, orbit_xp, orbit_yp, orbit_zp)
    orbit_xdot, orbit_ydot, orbit_zdot = rotate(ra_com_i, dec_com_i, orbit_xdotp, orbit_ydotp, orbit_zdotp)
    orbit_x_ref, orbit_y_ref, orbit_z_ref = rotate(ra_com_i, dec_com_i, orbit_xp_ref, orbit_yp_ref,
                                                   orbit_zp_ref)
    x_f = x_i + (dx_sys * del_epochs) + (orbit_x - orbit_x_ref)
    y_f = y_i + (dy_sys * del_epochs) + (orbit_y - orbit_y_ref)
    z_f = z_i + (dz_sys * del_epochs) + (orbit_z - orbit_z_ref)

    dx_f = dx_sys + orbit_xdot
    dy_f = dy_sys + orbit_ydot
    dz_f = dz_sys + orbit_zdot

    ra_f, dec_f, parallax_f, pmra_f, pmdec_f, rv_f = xyz2radecz(x_f, y_f, z_f, dx_f, dy_f, dz_f)

    return ra_f, dec_f, parallax_f, pmra_f, pmdec_f, rv_f


def light_timing_correction_iteration(epochs1, distances1, distances2):
    """
    Calculates the updated epochs at which orbital or proper motion values should be calculated
    to correct for light timing delays.

    Args:
        epochs1 (array): epochs at which values have already been calculated (MJD)
        distances1 (array): distances assumed when calculating values at epochs1 (pc)
        distances2 (array): new distances calculated from epochs1

    Returns:
        epochs2 (array): epochs corrected for light travel time between distances1 and distances2
    """
    c = 0.00083942887
    del_d = (distances2 - distances1) * u.pc
    del_t = del_d / const.c
    epochs2 = epochs1 + del_t.to(u.day).value
    return epochs2


def light_timing_epoch_correction(epochs, ra_com_i, dec_com_i, plx_com_i, pmra_com_i, pmdec_com_i, rv_com_i,
                                  ref_epoch):
    my206265 = 180 / np.pi * 60 * 60
    d0 = 1 / np.tan(plx_com_i / 1000 / my206265) / my206265
    ra1, dec1, parallax1, pmra1, pmdec1, rv1 = transform_motion(epochs, ra_com_i, dec_com_i,
                                                                plx_com_i, pmra_com_i, pmdec_com_i,
                                                                rv_com_i, 0, 0, 0, 0, 0, 0, ra_com_i,
                                                                dec_com_i, ref_epoch, 0, 0, 0)

    d1 = 1 / np.tan(parallax1 / 1000 / my206265) / my206265

    corrected_epochs = light_timing_correction_iteration(epochs, d0, d1)

    return corrected_epochs


def calc_corrected_orbit(sma, ecc, inc, argp, lan, tau, plx, mtot, m0, tau_ref_epoch, epochs, ref_epoch,
                         ra_com_i, dec_com_i, plx_com_i, pmra_com_i, pmdec_com_i, rv_com_i, mass, n_orbits):

    raoff, decoff, zoff, radot, decdot, zdot = kepler.calc_orbit(
        np.append(epochs, ref_epoch), sma, ecc, inc, argp, lan, tau, plx, mtot,
        mass_for_Kamp=m0, tau_ref_epoch=tau_ref_epoch,
        return_cartesian=True
    )

    x_com_i, y_com_i, z_com_i, dx_com_i, dy_com_i, dz_com_i = radecz2xyz(ra_com_i, dec_com_i,
                                                                         plx_com_i, pmra_com_i,
                                                                         pmdec_com_i, rv_com_i)
    n_epochs = len(epochs)

    orbit_xp = np.zeros((n_epochs + 1, 2, n_orbits))
    orbit_yp = np.zeros((n_epochs + 1, 2, n_orbits))
    orbit_zp = np.zeros((n_epochs + 1, 2, n_orbits))
    orbit_xdotp = np.zeros((n_epochs + 1, 2, n_orbits))
    orbit_ydotp = np.zeros((n_epochs + 1, 2, n_orbits))
    orbit_zdotp = np.zeros((n_epochs + 1, 2, n_orbits))

    orbit_xp[:, 0, :] = np.reshape(raoff, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_yp[:, 0, :] = np.reshape(decoff, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_zp[:, 0, :] = np.reshape(zoff, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_xp[:, 1, :] = np.reshape(raoff, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot
    orbit_yp[:, 1, :] = np.reshape(decoff, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot
    orbit_zp[:, 1, :] = np.reshape(zoff, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot

    orbit_xdotp[:, 0, :] = np.reshape(radot, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_ydotp[:, 0, :] = np.reshape(decdot, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_zdotp[:, 0, :] = np.reshape(zdot, (n_epochs + 1, n_orbits)) * -1 * mass / mtot
    orbit_xdotp[:, 1, :] = np.reshape(radot, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot
    orbit_ydotp[:, 1, :] = np.reshape(decdot, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot
    orbit_zdotp[:, 1, :] = np.reshape(zdot, (n_epochs + 1, n_orbits)) * (mtot - mass) / mtot

    orbit_x_ref, orbit_y_ref, orbit_z_ref = rotate(ra_com_i, dec_com_i, orbit_xp[-1, 0, 0],
                                                   orbit_yp[-1, 0, 0], orbit_zp[-1, 0, 0])

    ra_A_i, dec_A_i, plx_A_i, _, _, _ = xyz2radecz(orbit_x_ref + x_com_i, orbit_y_ref + y_com_i,
                                                   orbit_z_ref + z_com_i, 0, 0, 0)

    ra_A, dec_A, parallax_A, pmra_A, pmdec_A, rv_A = transform_motion(epochs, ra_A_i, dec_A_i,
                                                                      plx_A_i, pmra_com_i, pmdec_com_i,
                                                                      rv_com_i,
                                                                      orbit_xdotp[:-1, 0, 0],
                                                                      orbit_ydotp[:-1, 0, 0],
                                                                      orbit_zdotp[:-1, 0, 0],
                                                                      orbit_xp[:-1, 0, 0], orbit_yp[:-1, 0, 0],
                                                                      orbit_zp[:-1, 0, 0], ra_com_i,
                                                                      dec_com_i, ref_epoch,
                                                                      orbit_xp[-1, 0, 0],
                                                                      orbit_yp[-1, 0, 0],
                                                                      orbit_zp[-1, 0, 0])

    orbit_x_ref, orbit_y_ref, orbit_z_ref = rotate(ra_com_i, dec_com_i, orbit_xp[-1, 1, 0],
                                                   orbit_yp[-1, 1, 0], orbit_zp[-1, 1, 0])
    ra_B_i, dec_B_i, plx_B_i, _, _, _ = xyz2radecz(orbit_x_ref + x_com_i, orbit_y_ref + y_com_i,
                                                   orbit_z_ref + z_com_i, 0, 0, 0)

    ra_B, dec_B, parallax_B, pmra_B, pmdec_B, rv_B = transform_motion(epochs, ra_B_i, dec_B_i,
                                                                      plx_B_i, pmra_com_i, pmdec_com_i,
                                                                      rv_com_i,
                                                                      orbit_xdotp[:-1, 1, 0], orbit_ydotp[:-1, 1, 0],
                                                                      orbit_zdotp[:-1, 1, 0],
                                                                      orbit_xp[:-1, 1, 0], orbit_yp[:-1, 1, 0],
                                                                      orbit_zp[:-1, 1, 0], ra_com_i,
                                                                      dec_com_i, ref_epoch,
                                                                      orbit_xp[-1, 1, 0],
                                                                      orbit_yp[-1, 1, 0],
                                                                      orbit_zp[-1, 1, 0])

    corrected_raoff = (ra_B - ra_A) * np.cos(np.radians(dec_A))
    corrected_decoff = dec_B - dec_A
    corrected_vz_i = rv_B

    return corrected_raoff, corrected_decoff, corrected_vz_i
