#!/usr/bin/env python
# coding: utf-8

# # Homework 7
# ## 100 points total
# Due March. 13th, 2025 at 3:30 pm
# 
# ### Instructions
# <font color='red'>Please read the following instructions carefully! </font> 
# * All plots generated must have an xlabel, a ylabel, a legend, and a caption that interprets the plot.
# 
# * Axis limits must be sensible, meaning that you may need to set the axis limits via kwargs "xlim" and "ylim" in order to make your plot more interpretable. 
# 
# * To create a caption, add a new cell below a plot by using the option "Insert > Insert Cell below", then change the cell into a markdown cell by using "Cell > Cell Type > Markdown".
# 
# * Need help making a matrix or equation in LaTeX? See the [Piazza resources](https://piazza.com/ucsd/winter2022/beng123/resources)  for LaTeX!
# * Use ``print`` statements when necessary to display your answer!
# 
# * For this homework, all plots must be generated **WITHOUT** using the ``view_time_profile`` or ``view_tiled_phase_portraits`` MASSpy methods.
# 
# ### Submission
# This submission has **one** component: 
# * (1) iPython notebook with typed and coded solutions.

# ## Coding questions (100 points)
# <font color='red'>All code must run from top to bottom without error. To test your notebook, use the option "Kernel > Restart and Run all"</font> 
# 
# 
# For help loading models from JSON files, see the tutorial on [Reading and Writing Models](https://masspy.readthedocs.io/en/latest/tutorials/reading_writing_models.html#JSON) in the documentation.
# 
# ### Import packages

# In[2]:


from mass.io.json import load_json_model
from mass import Simulation, strip_time
from mass.visualization import plot_time_profile, plot_phase_portrait

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


# ### (1) Reconstruction of the Red Blood Cell (RBC) metabolic model (40 points)
# 
# **(a)** Import the glycolysis, pentose phosphate pathway, AMP metabolism, and hemoglobin models from the provided JSON files in the `models` directory.

# In[3]:


from os.path import join

glycolysis = load_json_model(join("models", "Glycolysis.json"))
PPP = load_json_model(join("models", "PentosePhosphatePathway.json"))
AMP = load_json_model(join("models", "AMPSalvageNetwork.json"))
hemoglobin = load_json_model(join("models", "Hemoglobin.json"))


# **(b)** Form the RBC network correctly by [integrating the glycolysis, pentose phosphate pathway and AMP metabolism modules](https://masspy.readthedocs.io/en/latest/education/sb2/chapters/sb2_chapter12.html#Network-Integration). (2 points)
# 
# Remove the 6 unnecessary boundary reactions and 5 unnecessary boundary conditions. (5 points)
# 
# Briefly discuss why these reactions/conditions can be removed in 2-5 sentences. (3 points)

# In[4]:


fullppp = glycolysis.merge(PPP, inplace = False)
fullppp.id = "Full_PPP"
fullppp.remove_reactions([
    r for r in fullppp.boundary
    if r.id in ["SK_g6p_c", "DM_f6p_c", "DM_g3p_c", "DM_r5p_c"]])
fullppp.remove_boundary_conditions(["g6p_b", "f6p_b", "g3p_b", "r5p_b"])
core_network = fullppp.merge(AMP, inplace = False)
core_network.id = "Core_Model"
core_network.remove_reactions([
    r for r in core_network.boundary
    if r.id in ["DM_amp_c", "SK_amp_c"]])
core_network.remove_boundary_conditions(["amp_b"])


# These reactions/conditions should be removed in order to prevent integration issues from arising. Integration issues arise when common nodes and fluxes overlap. Hence, removing certain nodes and reactions simplifes the integrated model and prevents issues from obstrcuting the simulation of the network. 

# **(c)** Adjust the stoichiometry of a reaction in the AMP Salvage Network. Integrate hemoglobin module into the RBC model. Set the model ID as "RBC" and display the correct model overview. (5 points)

# In[5]:


core_network.reactions.PRPPS.subtract_metabolites({
    core_network.metabolites.atp_c: -1,
    core_network.metabolites.adp_c: 2})
core_network.reactions.PRPPS.add_metabolites({
    core_network.metabolites.amp_c: 1})
RBC = core_network.merge(hemoglobin, inplace = False)
RBC.id = "RBC"
RBC


# **(d)** Define the steady state fluxes for the merged model. Display the correct steady state flux map, the equilibrium constants, and the computed PERCs values an organized DataFrame, where column names are the reactions. (15 points)
# 
# __Hint 1:__ Independent fluxes values can be found in the textbook
# 
# __Hint 2:__ Make sure the order of reactions in your model matches the order of the MinSpan Pathways! The CSV file containing the pathways can be read as follows:
# 
# >minspan_pathways = pd.read_csv('minspan_pathways.csv', index_col=0)
# 
# Values can be extracted from the DataFrame using the values attribute.
# 
# >minspan_pathways = minspan_pathways.values
# 
# __Hint 3:__ Do not calculate PERCs for "ADK1", "SK_o2_c", "HBDPG", "HBO1", "HBO2", "HBO3", "HBO4". Set PERCs for hemoglobin binding and the oxygen exchange manually as follows:
# 
# model.update_parameters({
#     "kf_SK_o2_c": 509726,
#     "kf_HBDPG": 519613, 
#     "kf_HBO1": 506935, 
#     "kf_HBO2": 511077, 
#     "kf_HBO3": 509243, 
#     "kf_HBO4": 501595
# })

# In[6]:


import sympy as sym
INF = float("inf")

minspan_pathways = pd.read_csv('minspan_pathways.csv', index_col = 0)
minspan_pathways = minspan_pathways.values

reaction_ids = [r.id for r in RBC.reactions]

independent_fluxes = {
    RBC.reactions.SK_glc__D_c: 1.12,
    RBC.reactions.DM_nadh: 0.2*1.12,
    RBC.reactions.GSHR : 0.42,
    RBC.reactions.SK_ade_c: -0.014,
    RBC.reactions.ADA: 0.01,
    RBC.reactions.SK_adn_c: -0.01,
    RBC.reactions.ADNK1: 0.12,
    RBC.reactions.SK_hxan_c: 0.097,
    RBC.reactions.DPGM: 0.441}

ssfluxes = RBC.compute_steady_state_fluxes(
    minspan_pathways,
    independent_fluxes,
    update_reactions = True)

RBC.update_parameters({ "kf_SK_o2_c": 509726, "kf_HBDPG": 519613, "kf_HBO1": 506935, 
                         "kf_HBO2": 511077, "kf_HBO3": 509243, "kf_HBO4": 501595 })

percs = RBC.calculate_PERCs(
    fluxes={
        r: flux for r, flux in RBC.steady_state_fluxes.items()
        if r.id not in [
            "ADK1", "SK_o2_c", "HBDPG", "HBO1", "HBO2", "HBO3", "HBO4"]},
    update_reactions=True)

value_dict = {sym.Symbol(str(met)): ic
              for met, ic in RBC.initial_conditions.items()}
value_dict.update({sym.Symbol(str(met)): bc
                   for met, bc in RBC.boundary_conditions.items()})
table = []
for p_key in ["Keq", "kf"]:
    symbol_list, value_list = [], []
    for p_str, value in RBC.parameters[p_key].items():
        symbol_list.append(r"$%s_{\text{%s}}$" % (p_key[0], p_str.split("_", 1)[-1]))
        value_list.append("{0:.3f}".format(value) if value != INF else r"$\infty$")
        value_dict.update({sym.Symbol(p_str): value})
    table.extend([symbol_list, value_list])
    
table.append(["{0:.6f}".format(float(ratio.subs(value_dict)))
                   for ratio in strip_time(RBC.get_mass_action_ratios()).values()])
table.append(["{0:.6f}".format(float(ratio.subs(value_dict)))
                   for ratio in strip_time(RBC.get_disequilibrium_ratios()).values()])
table = pd.DataFrame(np.array(table).T, index=reaction_ids,
                          columns=[r"K_${eq}$ Symbol", r"$K_{eq}$ Value", "PERC Symbol",
                                   "PERC Value", r"$\Gamma$", r"$\Gamma/K_{eq}$"]).T
new_data = pd.DataFrame(list(ssfluxes.values()), index = reaction_ids,
                        columns=[r"$\textbf{v}_{\mathrm{stst}}$"]).T
table = pd.concat([table, new_data])

table


# **(e)** Graphically verify that the model is in steady state. Use a logarithmic x-axis and a logarithmic y-axis. (5 points)

# In[7]:


fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))

sim = Simulation(reference_model = RBC)
sim.find_steady_state(RBC, strategy = "simulate", update_values = True) 
conc_sol, flux_sol = sim.simulate(RBC, time = (0, 1000), verbose = True)

plt.xscale('log')
plt.yscale('log')

plot_time_profile(conc_sol, ax = ax1, legend = "right outside", 
                    xlabel = "Time", ylabel = "Concentration",
                    title = ("Time vs. Concentration Profile", {"size" : "large"}))


# According to the graph, the concentrations of all of the metabolites in the merged network remain constant over time. Eventually, it can be concluded that the merged model is in steady state. 

# ### (2) The RBC network with and without PFK (60 points)
# 
# **(a)** Form a steady state model of the RBC with the PFK enzyme module.(10 points)
# 
# > Import the simplified PFK enzyme module from the provided JSON file (model based on [Chapter 14](https://masspy.readthedocs.io/en/latest/education/sb2/chapters/sb2_chapter14.html)). Integrate the PFK module into a **copy** of the RBC model from Problem (1). Display the model overview.
# 
# __Hint:__ Do you need to remove any reactions?
# 
# > In a new cell, update the model to steady state conditions through simulation. Graphically verify that the model is in steady state. Use a logarithmic x-axis and a logarithmic y-axis.

# In[11]:


from os.path import join

PFK = load_json_model(join("models", "PFK.json"))
RBC_copy = RBC.copy()
RBC_PFK = RBC_copy.merge(PFK, inplace = False)
RBC_PFK.id = "RBC_PFK"
RBC_PFK.remove_reactions([RBC_PFK.reactions.PFK])
RBC_PFK


# In[12]:


fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))

sim = Simulation(reference_model = RBC_PFK)
sim.find_steady_state(RBC_PFK, strategy = "simulate", update_values = True) 
conc_sol, flux_sol = sim.simulate(RBC_PFK, time = (0, 1000), verbose = True)

plt.xscale('log')
plt.yscale('log')

plot_time_profile(conc_sol, ax = ax1, legend = "right outside", 
                    xlabel = "Time", ylabel = "Concentration",
                    title = ("Time vs. Concentration Profile", {"size" : "large"}))


# **(b)** Apply a 20% increase in the rate of ATP utilization. Compare the RBC without the PFK module and the RBC with the PFK module. (20 points)
# 
# Plot a time profile of the flux through PFK. (10 points)
# 
# Briefly interpret the results in terms of the PFK regulatory mechanism. 
# 
# __Hint 1:__ Make an aggregate of all form of PFK in order to compare RBC and RBC_PFK
# 
# Explain the results in terms of the PFK regulatory mechanism. (20 points)
# 
# __Hint 2:__ See the section of the documentation with PFK with RBC metabolic network for an example.
# 

# In[32]:


fig, (ax1) = plt.subplots(nrows = 1, ncols = 1, figsize = (5,5))

PFK_module = RBC_PFK.enzyme_modules.PFK

equations_dict = {}
for i in range(0, 5):
    PFK_rxns = ["PFK_R{0:d}{1:d}".format(i, j) for j in range(1, 4)]
    equations_dict["R{0:d}i".format(i)] = [" + ".join(PFK_rxns),
                                           PFK_rxns]
    
catalyzation_rxns = PFK_module.enzyme_module_reactions_categorized.get_by_id("PFK_catalyzation")
catalyzation_rxns = [m.id for m in catalyzation_rxns.members]

equations_dict["PFK"] = [" + ".join(catalyzation_rxns),
                         catalyzation_rxns]
    
for i, model, in enumerate([RBC, RBC_PFK]):
    sim = Simulation(model)
    flux_solution = sim.simulate(
        model, time=(0, 1000),
        perturbations={"kf_ATPM": "kf_ATPM * 1.20"})[1]
    
    if i > 0:
        flux_solution.make_aggregate_solution(
            "PFK", equation=equations_dict["PFK"][0],
            variables=equations_dict["PFK"][1])

    plot_time_profile(
        flux_solution, observable = "PFK", ax=ax1,
        legend = [model.id, "right outside"], plot_function="semilogx",
        xlabel="Time", ylabel="Flux",
        title=("Net Flux through PFK", {"size" : "large"}))


# According to the graphs, an increase in ATP utilization disrupts the steady state of the fluxes in both the RBC model without PFK and the RBC model with PFK. However, the flux decay takes longer in the RBC model without PFK. A higher ATP utilization rate indicates a higher concentration of ATP. PFK is inhibited by high concentrations of ATP and activated by AMP. Hence, the absence of PFK in the RBC model without PFK gives more time for the ATP utilization flux to decay as the high concentration of ATP gradually decreases over time and is consumed without the help of an enzyme. Once ATP is regenerated, the flux gradually rises again until it returns back to its original state. On the other hand, in the model with PFK, the high concentration of ATP increases the flux, or PFK activity, initially. However, as ATP molecules start inhibiting the PFK molecules allosterically, the ATP utilization flux decreases sharply and reaches its lowest point earlier than the original RBC model without PFK. Once AMP becomes increasingly available, though, the PFK molecules are reactivated and the ATP utilization flux starts increasing again until its original state is restored. Thus, the regulatory mechanism of PFK differentiates a model without its presence from a model with its presence by influencing ATP utilization rates. 
