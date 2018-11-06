'''
NETPYNE simulator compliant export for:

Components:
    RS (Type: izhikevich2007Cell:  v0=-0.06 (SI voltage) k=7.0E-7 (SI conductance_per_voltage) vr=-0.06 (SI voltage) vt=-0.04 (SI voltage) vpeak=0.035 (SI voltage) a=30.0 (SI per_time) b=-2.0E-9 (SI conductance) c=-0.05 (SI voltage) d=1.0E-10 (SI current) C=1.0E-10 (SI capacitance))
    RS_Iext (Type: pulseGenerator:  delay=0.0 (SI time) duration=0.52 (SI time) amplitude=1.0E-10 (SI current))
    net1 (Type: network)
    sim1 (Type: Simulation:  length=0.52 (SI time) step=1.0E-6 (SI time))


    This NETPYNE file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.5.4
         org.neuroml.model   v1.5.4
         jLEMS               v0.9.9.1

'''
# Main NetPyNE script for: net1

# See https://github.com/Neurosim-lab/netpyne

from netpyne import specs  # import netpyne specs module
from netpyne import sim    # import netpyne sim module

from neuron import h

import sys


###############################################################################
# NETWORK PARAMETERS
###############################################################################

nml2_file_name = 'NET_2007One.net.nml'

###############################################################################
# SIMULATION PARAMETERS
###############################################################################

simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

# Simulation parameters
simConfig.duration = simConfig.tstop = 520.0 # Duration of the simulation, in ms
simConfig.dt = 0.001 # Internal integration timestep to use

# Seeds for randomizers (connectivity, input stimulation and cell locations)
# Note: locations and connections should be fully specified by the structure of the NeuroML,
# so seeds for conn & loc shouldn't affect networks structure/behaviour
simConfig.seeds = {'conn': 0, 'stim': 123456789, 'loc': 0} 

simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
simConfig.verbose = False  # show detailed messages 

# Recording 
simConfig.recordCells = ['all']  
simConfig.recordTraces = {}
simConfig.saveCellSecs=False
simConfig.saveCellConns=False
simConfig.gatherOnlySimData=True 

# For saving to file: exIzh.dat (ref: of0)
# Column: v: Pop: RS_pop; cell: 0; segment id: $oc.segment_id; segment name: soma; Neuron loc: soma(0.5); value: v (v)
simConfig.recordTraces['of0_RS_pop_0_soma_v'] = {'sec':'soma','loc':0.5,'var':'v','conds':{'pop':'RS_pop','cellLabel':0}}



simConfig.plotCells = ['all']


simConfig.recordStim = True  # record spikes of cell stims
simConfig.recordStep = simConfig.dt # Step size in ms to save data (eg. V traces, LFP, etc)



# Analysis and plotting, see http://neurosimlab.org/netpyne/reference.html#analysis-related-functions
simConfig.analysis['plotRaster'] = False  # Plot raster
simConfig.analysis['plot2Dnet'] = False  # Plot 2D net cells and connections
simConfig.analysis['plotSpikeHist'] = False # plot spike histogram
simConfig.analysis['plotConn'] = False # plot network connectivity
simConfig.analysis['plotSpikePSD'] = False # plot 3d architecture

# Saving
simConfig.filename = 'net1.txt'  # Set file output name
simConfig.saveFileStep = simConfig.dt # step size in ms to save data to disk
# simConfig.saveDat = True # save to dat file


###############################################################################
# IMPORT & RUN
###############################################################################

print("Running a NetPyNE based simulation for %sms (dt: %sms) at %s degC"%(simConfig.duration, simConfig.dt, simConfig.hParams['celsius']))

gids = sim.importNeuroML2SimulateAnalyze(nml2_file_name,simConfig)

print("Finished simulation")


###############################################################################
#   Saving data (this ensures the data gets saved in the format/files 
#   as specified in the LEMS <Simulation> element)
###############################################################################


if sim.rank==0: 
    print("Saving traces to file: exIzh.dat (ref: of0)")

 
    # Column: t
    col_of0_t = [i*simConfig.dt for i in range(int(simConfig.duration/simConfig.dt))]

    # Column: v: Pop: RS_pop; cell: 0; segment id: $oc.segment_id; segment name: soma; value: v
    col_of0_v = sim.allSimData['of0_RS_pop_0_soma_v']['cell_%s'%gids['RS_pop'][0]]

    dat_file_of0 = open('exIzh.dat', 'w')
    for i in range(len(col_of0_t)):
        dat_file_of0.write( '%s\t'%(col_of0_t[i]/1000.0) +  '%s\t'%(col_of0_v[i]/1000.0) +  '\n')
    dat_file_of0.close()




    print("Saved all data.")

if '-nogui' in sys.argv:
    quit()
