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

    def __init__(self, n_ly, dx_array, dy_array, dz_array,
                 perm_h, perm_v, poro, hcap, tcond,
                 well_idx_df, depth_to_top, geothermal_grad,
                 ts_mult, ts_max, n_points=100):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        self.platform = 'cpu' # Does 'gpu' also work?
        # #------------Set Node Number------------
        self.n_ly_cap = int(n_ly[0])  # layer number in cap rock
        self.n_ly_res = int(n_ly[1]) # layer number in reservoir rock
        self.n_ly_bttm = int(n_ly[2])  # layer number in bottom formation

        nx = len(dx_array)
        ny = len(dy_array)
        nz = len(dz_array)

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz,
                                         dx=dx_array, dy=dy_array, dz=dz_array,
                                         permx=perm_h, permy=perm_h, permz=perm_v,
                                         poro=poro, start_z=depth_to_top, hcap=hcap, rcond=tcond)

        self.wells_def = well_idx_df

        self.reservoir.boundary_volumes['yz_minus'] = 1e20
        self.reservoir.boundary_volumes['yz_plus'] = 1e20
        self.reservoir.boundary_volumes['xz_minus'] = 1e20
        self.reservoir.boundary_volumes['xz_plus'] = 1e20

        self.geothermal_grad = geothermal_grad

        # ------------create pre-defined physics for geothermal-------------
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points, 0.1, 150, 500, 7500, cache=False)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()
        # self.set_physics_super(zero=1e-12, n_points=n_points, components=["H2O","C1"])

        self.set_sim_params(first_ts=1e-3, mult_ts=ts_mult, max_ts=ts_max, runtime=3650, tol_newton=1e-4, tol_linear=1e-8,
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
        btm_ind = self.n_ly_cap + self.n_ly_res - 1

        well_counters = {"H": 0, "C": 0}

        for _, w in self.wells_def.iterrows():
            wtype = w['type'].upper()
            ix = int(w['well_index_x'])
            iy = int(w['well_index_y'])

            if wtype not in well_counters:
                raise ValueError(
                    f"Unknown well type '{wtype}'. Expected 'H' or 'C'."
                )

            well_counters[wtype] += 1
            well_name = f"{wtype}{well_counters[wtype]}"

            self.reservoir.add_well(well_name)

            for k in range(top_ind, btm_ind + 1):
                self.reservoir.add_perforation(
                    well_name,
                    cell_index=(ix, iy, k),
                    verbose=True,
                    multi_segment=False,
                    well_indexD=0,
                    skin = 10
                )

    def set_initial_conditions(self):
        input_depth = [0., np.amax(self.reservoir.mesh.depth)]
        input_distribution = {'pressure': [1., 1. + input_depth[1] * 100. / 1000],
                              'temperature': [283.15, 283.15 + input_depth[1] * self.geothermal_grad / 1000]}
        return self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                    input_distribution=input_distribution,
                                                                    input_depth=input_depth)
    def set_well_controls(self, h_func=None, l_func=None):
        for i, w in enumerate(self.reservoir.wells):
            if 'H' in w.name:
                self.physics.set_well_controls(wctrl=w.control,control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True, target=0., phase_name='water', inj_composition=[],
                                               inj_temp=273.15 + 90)
            elif 'C' in w.name:
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
            if 'C' in w.name:
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

