# -*- coding: utf-8 -*-
"""
Joanmarie Del Vecchio's periglacial diffuser <3
Built on the skeleton of Rachel Glade's DepthDependentDiffuser
"""

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import figure, show, plot, xlabel, ylabel, title
import copy

## Import Landlab components and utilities
from landlab.plot import imshow_grid
from landlab import RasterModelGrid, CLOSED_BOUNDARY, CORE_NODE, FIXED_VALUE_BOUNDARY, INACTIVE_LINK, Component
from landlab.io import read_esri_ascii

#++++ KW
# load permamodel ...
from permamodel.components import KuFlex_method
from permamodel.components import bmi_KuFlex_component

kuflex = bmi_KuFlex_component.BmiKuFlexMethod()

# %%

class PeriglacialDiffuser(Component):


    _name = "PeriglacialDiffuser"

    _input_var_names = (
        "topographic__elevation",
        "soil__depth",
        "soil_production__rate",
    )

    _output_var_names = (
        "soil__flux",
        "topographic__slope",
        "topographic__elevation",
        "bedrock__elevation",
        "soil__depth",
    )

    _var_units = {
        "topographic__elevation": "m",
        "topographic__slope": "m/m",
        "soil__depth": "m",
        "soil__flux": "m^2/yr",
        "soil_production__rate": "m/yr",
        "bedrock__elevation": "m",
    }

    _var_mapping = {
        "topographic__elevation": "node",
        "topographic__slope": "link",
        "soil__depth": "node",
        "soil__flux": "link",
        "soil_production__rate": "node",
        "bedrock__elevation": "node",
    }

    _var_doc = {
        "topographic__elevation": "elevation of the ground surface",
        "topographic__slope": "gradient of the ground surface",
        "soil__depth": "depth of soil/weather bedrock",
        "soil__flux": "flux of soil in direction of link",
        "soil_production__rate": "rate of soil production at nodes",
        "bedrock__elevation": "elevation of the bedrock surface",
    }

    def __init__(self, grid, linear_diffusivity=(0.02/365)):
        """Initialize the PeriglacialDiffuser."""
        #I don't know how functions work like where to define things
        #so I have linear diffusivity up there 
        #Also everything is divided by 365 since I have daily timesteps
        # Store grid and parameters
        self._grid = grid
        self.K = linear_diffusivity
        #self.soil_transport_decay_depth = soil_transport_decay_depth

        # create fields
        qs_gel = np.zeros(1)
        qs_heave = np.zeros(1)

        #qs for each process is recorded each day
        
        # elevation
        if "topographic__elevation" in self.grid.at_node:
            self.elev = self.grid.at_node["topographic__elevation"]
        else:
            self.elev = self.grid.add_zeros("node", "topographic__elevation")

        # slope
        if "topographic__slope" in self.grid.at_link:
            self.slope = self.grid.at_link["topographic__slope"]
        else:
            self.slope = self.grid.add_zeros("link", "topographic__slope")

        # soil depth
        if "soil__depth" in self.grid.at_node:
            self.depth = self.grid.at_node["soil__depth"]
        else:
            self.depth = self.grid.add_zeros("node", "soil__depth")

        # soil flux
        if "soil__flux" in self.grid.at_link:
            self.flux = self.grid.at_link["soil__flux"]
        else:
            self.flux = self.grid.add_zeros("link", "soil__flux")

        # weathering rate
        if "soil_production__rate" in self.grid.at_node:
            self.soil_prod_rate = self.grid.at_node["soil_production__rate"]
        else:
            self.soil_prod_rate = self.grid.add_zeros("node", "soil_production__rate")

        # bedrock elevation
        if "bedrock__elevation" in self.grid.at_node:
            self.bedrock = self.grid.at_node["bedrock__elevation"]
        else:
            self.bedrock = self.grid.add_zeros("node", "bedrock__elevation")

    def soilflux(self, dt):
        """Calculate soil flux for a time period 'dt'.

        Parameters
        ----------

        dt: float (time)
            The imposed timestep.
        """

        # update soil thickness
        self.grid.at_node["soil__depth"][:] = (
            self.grid.at_node["topographic__elevation"]
            - self.grid.at_node["bedrock__elevation"]
        )

        # Calculate soil depth at links.
        H_link = self.grid.map_value_at_max_node_to_link(
            "topographic__elevation", "soil__depth"
        )

        # Calculate gradients
        slope = self.grid.calc_grad_at_link(self.elev)
        slope[self.grid.status_at_link == INACTIVE_LINK] = 0.0
        
#        # Once soil depth is calculated, initialize thermal domain
#        # Ok, right now I'm going to calculate a global thermal profile
#        # for all soils. This is fine for now. 
#        
        #These were imposed in Bovy, I chose my own to fit field data
        gel_exp = 1.0
        creep_exp = 1.2
        
        #++++++ KW
        # make simple input from topo:
        
        dump_tm = self.grid.at_node["topographic__elevation"]/100.*0.65-9.0
        dump_tm = np.reshape(dump_tm, [53088,1])
        print(dump_tm.shape, dump_tm.max())
        
        #++++++ KW
        # Run Ku to get ALT:
        
        kuflex.initialize('KuFlex_method.cfg')
        
        kuflex.set_value('atmosphere_bottom_air__temperature', dump_tm)
        
        kuflex.update()
        
        h_a = kuflex.get_value('soil__active_layer_thickness')
        
        print(np.nanmean(h_a), h_a.shape)
        
        kuflex.finalize()
        
        #++++++
        
        #Have an array of depths at which temperature is calculated
        H_therm = np.linspace(0.05, 5.0, num=100)
        T_H = np.zeros((len(H_therm)))
        
        print(len(H_link))

        for t in range(365):
                    #print(t)        
                    T_c = (t + 1) / 365.0 
                    for i in range(len(H_therm)):
                        T_H[i] = T_m - (T_a * np.cos((2. * np.pi *T_c) - (H_therm[i]/h_T)) * np.exp(-H_therm[i]/h_T))
                        # Bovy formulation. Would be nice to use Eqn 10 in Gold + Lachenbruch to do spatially variable ALT
                        # thermal properties of soil, rock etc. could change based on saturation, density, many things, yay
        
                        # this loop is to find the 0C depth (ALT) because I can't rearrange equations
                        i=0
                        for i in range(len(H_therm)):
                            if T_H[1] < 0:
                                h_a = 0
                                too_warm = 0
                                break
        
                            elif (T_H[i]) - .00001 < 0.0001:
                                h_a = (H_therm[i])
                                too_warm = 0
                                break
        
                            else:
                                #h_a = H_link #THIS IS YOUR PROBLEM!!!
                                too_warm = 1
                        
                    if too_warm == 1:
                         #self.flux[:] = ((-self.K ** creep_exp) * slope * h_star_heave * (1 -(np.exp((-H_link / h_star_heave)))))
                        self.flux[:] = (-self.K ** creep_exp) * slope * (H_link ** creep_exp)                   
                    else:
                                                     
                        if h_a > 0. and too_warm == 0: 
                             self.flux[:] = ((-self.K ** gel_exp) * slope * h_a) + (-self.K ** creep_exp) * slope * (H_link ** creep_exp)
                        else:
                            self.flux[:] = 0.

                        
                    
#        V_gel_sum = (np.sum(V_gel, axis=1))             # sum annual movement at each depth
#        V_heave_sum = (np.sum(V_heave, axis=1))
#
#        qs_gel_sum = (np.sum(qs_gel))             # sum annual movement at each depth
#        qs_heave_sum = (np.sum(qs_heave))
#
#        cum_V = np.sum(V_heave_sum+V_gel_sum)-np.cumsum(V_heave_sum+V_gel_sum)
#        qs_sum = qs_gel_sum + qs_heave_sum


        # Calculate flux
#        self.flux[:] = (
#            -self.K
#            * slope
#            * self.soil_transport_decay_depth
#            * (1.0 - np.exp(-H_link / self.soil_transport_decay_depth))
#        )

                    # Calculate flux divergence
                    dqdx = self.grid.calc_flux_div_at_node(self.flux)
            
                    # Calculate change in soil depth
                    dhdt = self.soil_prod_rate - dqdx
            
                    # Calculate soil depth at nodes
                    self.depth[self.grid.core_nodes] += dhdt[self.grid.core_nodes] * dt
            
                    # prevent negative soil thickness
                    self.depth[self.depth < 0.0] = 0.0
            
                    # Calculate bedrock elevation
                    self.bedrock[self.grid.core_nodes] -= (
                        self.soil_prod_rate[self.grid.core_nodes] * dt
                    )
            
                    # Update topography
                    self.elev[self.grid.core_nodes] = (
                        self.depth[self.grid.core_nodes] + self.bedrock[self.grid.core_nodes]
                    )

    def run_one_step(self, dt):
        """

        Parameters
        ----------
        dt: float (time)
            The imposed timestep.
        """

        self.soilflux(dt)
        

# %%
#Read ASCII
(mg, z) = read_esri_ascii('t47_10m.asc', name='topographic__elevation')
mg.set_nodata_nodes_to_closed(z, -9999)
for edge in (mg.nodes_at_top_edge, mg.nodes_at_bottom_edge, mg.nodes_at_left_edge,
            mg.nodes_at_right_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY

#Plot ASCII
#imshow_grid(mg, 'topographic__elevation')

#copy elevation prior to diffusion (overridden after)
z0 = copy.deepcopy(z)
mg.at_node['topographic__elevation'] = copy.deepcopy(z0)

# Set soil thickness
soilTh = mg.add_zeros('node', 'soil__depth')
soilTh[:] = 1
mg.at_node['soil__depth'] = soilTh
# You must explicitly solve for bedrock or else the diffuser sets bedrock to 0!
mg.at_node['bedrock__elevation'] = (mg.at_node['topographic__elevation'] - mg.at_node['soil__depth'])

# Here's how tempearture works
T_m = 1 #MAT
T_a = 10. #amplitude
h_T = 0.7 # h_T encompasses thermal properties; selecting an h_T simplifies this

#Here's my temperatures when I was simulating warming, you can see I have the
#array boi commented out below cause I was just running it with constant temp
#T_m_array = np.linspace(-2, 0.001, num=100)

q=0

while q < 2:
    #start = time.time()
#    T_m = T_m_array[q]
    PGdiff = PeriglacialDiffuser(mg)
    PGdiff.soilflux(1)
    #end = time.time()
    #print("It took: ", end-start, " seconds to run!")
    #hollow[q,:]=(q, mg.at_node['topographic__elevation'][mg.find_nearest_node([1212224.0242403788, -708732.682404766])])
    print(q, mg.at_node['topographic__elevation'][mg.find_nearest_node([1212212.0242403788, -708727.682404766])])
    q +=1
    
imshow_grid(mg, (mg.at_node['topographic__elevation'] - z0))

