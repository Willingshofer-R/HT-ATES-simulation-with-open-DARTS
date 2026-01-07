from darts.discretizer import value_vector
from darts.physics.geothermal.physics import Geothermal
# from darts.physics.geothermal.property_container import PropertyContainerIAPWS
from darts.physics.geothermal.property_container import PropertyContainer
from darts.models.darts_model import DartsModel
from darts.engines import redirect_darts_output, well_control_iface, sim_params
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.eos_properties import EoSEnthalpy
from darts.physics.properties.flash import SinglePhase
from darts.physics.properties.viscosity import MaoDuan2009
from darts.physics.super.physics import Compositional
from darts.reservoirs.cpg_reservoir import CPG_Reservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.tools.gen_cpg_grid import gen_cpg_grid
from dartsflash.libflash import AQEoS
from dartsflash.components import CompData

redirect_darts_output('LogFile_Run_HT_ATES_DELFT.log')

from darts.engines import set_num_threads

set_num_threads(4)

import numpy as np


# %%
class Model(DartsModel):

    def __init__(self, n_points=100):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        self.platform = 'cpu'
        # #------------Set Node Number------------
        self.n_ly_cap = 0  # layer number in cap rock
        self.n_ly_res = 10  # layer number in reservoir rock
        self.n_ly_bttm = 0  # layer number in bottom formation

        self.nx = 50  # Grid number in x direction
        self.ny = 50  # Grid number in y direction
        self.nz = self.n_ly_cap + self.n_ly_res + self.n_ly_bttm  # Layer number in z direction

        self.dx = 5  # Grid interval in x direction. it is constant in this case
        self.dy = 5  # Grid interval in y direction. it is constant in this case
        self.dz = [15, 5,
                   15]  # Grid intervals in z layers. Each item in the array represents the layer interval from top to bottom.
        dz_list = np.concatenate([np.ones(self.n_ly_cap) * self.dz[0],
                                  np.ones(self.n_ly_res) * self.dz[1],
                                  np.ones(self.n_ly_bttm) * self.dz[2]])

        # ------------define layer permeabilities------------
        self.pmx = [500, 5000, 100]  # Permeability at the x direction in each layer
        self.pmy = [400, 5000, 70]  # Permeability at the y direction in each layer
        self.pmz = [50, 500, 10]  # Permeability at the z direction in each layer

        perm_x = np.concatenate([np.ones(self.nx * self.ny * self.n_ly_cap) * self.pmx[0],
                                 np.ones(self.nx * self.ny * self.n_ly_res) * self.pmx[1],
                                 np.ones(self.nx * self.ny * self.n_ly_bttm) * self.pmx[2]])

        perm_y = np.concatenate([np.ones(self.nx * self.ny * self.n_ly_cap) * self.pmy[0],
                                 np.ones(self.nx * self.ny * self.n_ly_res) * self.pmy[1],
                                 np.ones(self.nx * self.ny * self.n_ly_bttm) * self.pmy[2]])

        perm_z = np.concatenate([np.ones(self.nx * self.ny * self.n_ly_cap) * self.pmz[0],
                                 np.ones(self.nx * self.ny * self.n_ly_res) * self.pmz[1],
                                 np.ones(self.nx * self.ny * self.n_ly_bttm) * self.pmz[2]])

        # ------------define layer porosities------------
        self.poro = [0.2, 0.3, 0.15]  # Porosity in each layer

        poro = np.concatenate([np.ones(self.nx * self.ny * self.n_ly_cap) * self.poro[0],
                               np.ones(self.nx * self.ny * self.n_ly_res) * self.poro[1],
                               np.ones(self.nx * self.ny * self.n_ly_bttm) * self.poro[2]])

        self.depth_to_top = 100  # mesh lower bound coordinate by Z (the depth of the top layer) [m]

        # ------------discretize structured reservoir------------
        arrays = gen_cpg_grid(nx=self.nx, ny=self.ny, nz=self.nz,
                              dx=self.dx, dy=self.dy, dz=dz_list, start_z=self.depth_to_top,
                              permx=perm_x, permy=perm_y, permz=perm_z, poro=poro)

        nb = self.nx * self.ny * self.nz

        hcap = np.ones(nb)
        rcond = np.ones(nb)
        rcond[poro <= 0.1] = 2.2 * 86.4  # Shale conductivity kJ/m/day/K
        rcond[poro > 0.1] = 3 * 86.4  # Sandstone conductivity kJ/m/day/K
        hcap[poro <= 0.1] = 2300  # Shale heat capacity kJ/m3/K
        hcap[poro > 0.1] = 2450  # Sandstone heat capacity kJ/m3/K
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz,
                                      dx=self.dx, dy=self.dy, dz=dz_list,
                                         permx=perm_x, permy=perm_y, permz=perm_z,
                                         poro=poro, start_z=self.depth_to_top, hcap=hcap, rcond=rcond)

        self.reservoir.boundary_volumes['yz_minus'] = 1e20
        self.reservoir.boundary_volumes['yz_plus'] = 1e20
        self.reservoir.boundary_volumes['xz_minus'] = 1e20
        self.reservoir.boundary_volumes['xz_plus'] = 1e20
        # self.reservoir.set_boundary_volume(yz_minus=1e10, yz_plus=1e10, xz_minus=1e10, xz_plus=1e10)

        # ------------create pre-defined physics for geothermal------------
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points, 0.1, 150, 500, 7500, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()
        # self.set_physics_super(zero=1e-12, n_points=n_points, components=["H2O","C1"])

        self.set_sim_params(first_ts=1e-3, mult_ts=8, max_ts=30, runtime=3650, tol_newton=1e-4, tol_linear=1e-8,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))
        self.timer.node["initialization"].stop()

    def set_physics_super(self, zero, n_points, components, temperature=None):
        """Physical properties"""
        # Fluid components, ions and solid
        # phases = ["Aq", "V"]
        from darts.physics.super.property_container import PropertyContainer
        phases = ["Aq"]
        comp_data = CompData(components, setprops=True)

        aq_evaluators = {AQEoS.water: AQEoS.Jager2003,
                         AQEoS.solute: AQEoS.Ziabakhsh2012}
        aq = AQEoS(comp_data, aq_evaluators)

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, rock_comp=1e-5, min_z=zero / 10)

        # property_container.flash_ev = NegativeFlash2(flash_params)
        property_container.flash_ev = SinglePhase(nc=2)
        property_container.density_ev = dict([('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('Aq', MaoDuan2009(components))])
        property_container.rel_perm_ev = dict([('Aq', PhaseRelPerm("wat", swc=0.0))])

        property_container.enthalpy_ev = dict([('Aq', EoSEnthalpy(aq))])

        property_container.conductivity_ev = dict([('Aq', ConstFunc(172.8)), ])
        property_container.capillary_pressure_ev = ConstFunc(np.array([0.]))
        # property_container.rock_energy_ev = EnthalpyBasic(hcap=1.)  # hcap unit is kJ/kg/K

        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=100, min_z=zero / 10,
                                     max_z=1 - zero / 10, min_t=273.15 + 10, max_t=400,
                                     state_spec=Compositional.StateSpecification.PT, cache=False)
        self.physics.thermal = thermal
        self.physics.add_property_region(property_container)

        return

    def set_wells(self):
        top_ind = self.n_ly_cap + 1
        btm_ind = self.n_ly_cap + self.n_ly_res

        self.reservoir.add_well("H1")
        self.reservoir.add_well("L1")

        for i in range(top_ind, btm_ind + 1):
            self.reservoir.add_perforation("H1",
                                           cell_index=(int(self.nx / 2), int(self.ny * 3.5 / 5), i),
                                           verbose=True,
                                           multi_segment=False,
                                           well_indexD=0,
                                           skin=10)

            self.reservoir.add_perforation("L1",
                                           cell_index=(int(self.nx / 2), int(self.ny * 1.5 / 5), i),
                                           verbose=True,
                                           multi_segment=False,
                                           well_indexD=0,
                                           skin=10)


    def set_initial_conditions(self):
        input_depth = [0., np.amax(self.reservoir.mesh.depth)]
        input_distribution = {'pressure': [1., 1. + input_depth[1] * 100. / 1000],
                              'temperature': [283.15, 283.15 + input_depth[1] * 18. / 1000]}
        return self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                    input_distribution=input_distribution,
                                                                    input_depth=input_depth)
    def set_well_controls(self, h_func=None, l_func=None):
        for i, w in enumerate(self.reservoir.wells):
            if 'H' in w.name:
                self.physics.set_well_controls(wctrl=w.control,control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=0., phase_name='water', inj_composition=[],
                                               inj_temp=273.15 + 90)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False, target=0., phase_name='water')

    def set_rate_hot(self, rate, temp=300, func='inj'):
        for w in self.reservoir.wells:
            if 'H' in w.name:
                if func == 'inj':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=True, target=rate, phase_name='water', inj_composition=[],
                                                   inj_temp=temp)
                    # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
                if func == 'prod':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False, target=rate, phase_name='water')
                    # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)

    def set_rate_cold(self, rate, temp=300, func='inj'):
        # w = self.reservoir.wells[welln]
        for w in self.reservoir.wells:
            if 'L' in w.name:
                if func == 'inj':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=True, target=rate, phase_name='water', inj_composition=[],
                                                   inj_temp=temp)
                    # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
                if func == 'prod':
                    self.physics.set_well_controls(wctrl=w.control,
                                                   control_type=well_control_iface.VOLUMETRIC_RATE,
                                                   is_inj=False, target=rate, phase_name='water')
                    # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)

    # def set_well_controls(self, h_func=None, l_func=None):
    #     for i, w in enumerate(self.reservoir.wells):
    #         if 'H' in w.name:
    #             self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                            is_inj=True, target=0., phase_name='Aq',  inj_composition=[1 - 1e-12],
    #                                                inj_temp=273.15+90)
    #         else:
    #             self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                            is_inj=False, target=0., phase_name='Aq')
    #
    # def set_rate_hot(self, rate, temp=300, func='inj'):
    #     for w in self.reservoir.wells:
    #         if 'H' in w.name:
    #             if func == 'inj':
    #                 self.physics.set_well_controls(wctrl=w.control,
    #                                                control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                                is_inj=True, target=rate, phase_name='Aq', inj_composition=[1 - 1e-12],
    #                                                inj_temp=temp)
    #                 # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
    #             if func == 'prod':
    #                 self.physics.set_well_controls(wctrl=w.control,
    #                                                control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                                is_inj=False, target=rate, phase_name='Aq')
    #                 # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)
    #
    # def set_rate_cold(self, rate, temp=300, func='inj'):
    #     # w = self.reservoir.wells[welln]
    #     for w in self.reservoir.wells:
    #         if 'L' in w.name:
    #             if func == 'inj':
    #                 self.physics.set_well_controls(well=w, is_control=True,
    #                                                control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                                is_inj=True, target=rate, phase_name='Aq', inj_composition=[1 - 1e-12],
    #                                                inj_temp=temp)
    #                 # w.constraint = self.physics.new_bhp_water_inj(self.midrespress + self.bhp_limit, temp)
    #             if func == 'prod':
    #                 self.physics.set_well_controls(well=w, is_control=True,
    #                                                control_type=well_control_iface.VOLUMETRIC_RATE,
    #                                                is_inj=False, target=rate, phase_name='Aq')
    #                 # w.constraint = self.physics.new_bhp_prod(self.midrespress - self.bhp_limit)
