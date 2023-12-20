'''
NETPYNE simulator compliant export for:

Components:
    null (Type: notes)
    Ca_LVAst (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    K_Pst (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    K_Tst (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    Nap_Et2 (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    NaTa_t (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    SKv3_1 (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    SK_E2 (Type: ionChannelHH:  conductance=1.0E-11 (SI conductance))
    null (Type: notes)
    pas (Type: ionChannelPassive:  conductance=1.0E-11 (SI conductance))
    acell (Type: cell)
    pg0 (Type: pulseGenerator:  delay=0.25 (SI time) duration=0.5 (SI time) amplitude=4.5E-11 (SI current))
    ball_stick_net (Type: network)
    ballandstick_netpyne (Type: Simulation:  length=1.0 (SI time) step=2.5E-5 (SI time))


    This NETPYNE file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.10.0
         org.neuroml.model   v1.10.0
         jLEMS               v0.11.0

'''
# Main NetPyNE script for: ball_stick_net

# See https://github.com/Neurosim-lab/netpyne

from netpyne import specs  # import netpyne specs module
from netpyne import sim    # import netpyne sim module
from netpyne import __version__ as version

from neuron import h

import sys
import time
import datetime

class NetPyNESimulation():

    def __init__(self, tstop=1000.0, dt=0.025, seed=123456789, save_json=False):

        self.setup_start = time.time()
        

        ###############################################################################
        # NETWORK PARAMETERS
        ###############################################################################

        self.nml2_file_name = 'ballandstick.net.nml'

        ###############################################################################
        # SIMULATION PARAMETERS
        ###############################################################################

        self.simConfig = specs.SimConfig()   # object of class SimConfig to store the simulation configuration

        # Simulation parameters
        self.simConfig.duration = self.simConfig.tstop = tstop # Duration of the simulation, in ms
        self.simConfig.dt = dt # Internal integration timestep to use

        # Seeds for randomizers (connectivity, input stimulation and cell locations)
        # Note: locations and connections should be fully specified by the structure of the NeuroML,
        # so seeds for conn & loc shouldn't affect networks structure/behaviour
        self.simConfig.seeds = {'conn': 0, 'stim': 12345, 'loc': 0}

        self.simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
        self.simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
        self.simConfig.verbose = False  # show detailed messages
        
        # Recording
        self.simConfig.recordCells = ['all']
        self.simConfig.recordTraces = {}
        self.simConfig.saveCellSecs=False
        self.simConfig.saveCellConns=False
        self.simConfig.gatherOnlySimData=True

                # For saving to file: v.netpyne.dat (ref: output0)
                                
        # Column: soma_v: Pop: ball_stick_pop; cell: 0; segment id: 0; segment name: Seg0_soma0; Neuron loc: soma0(0.5); value: v (v)
        self.simConfig.recordTraces['output0_ball_stick_pop_0_Seg0_soma0_v'] = {'sec':'soma0','loc':0.5,'var':'v','conds':{'pop':'ball_stick_pop','cellLabel':0}}
                                
        # Column: dend_v1: Pop: ball_stick_pop; cell: 0; segment id: 1; segment name: Seg0_dend0; Neuron loc: dend0(0.108452566); value: v (v)
        self.simConfig.recordTraces['output0_ball_stick_pop_0_Seg0_dend0_v'] = {'sec':'dend0','loc':0.108452566,'var':'v','conds':{'pop':'ball_stick_pop','cellLabel':0}}
                                
        # Column: dend_v2: Pop: ball_stick_pop; cell: 0; segment id: 2; segment name: Seg1_dend0; Neuron loc: dend0(0.41267884); value: v (v)
        self.simConfig.recordTraces['output0_ball_stick_pop_0_Seg1_dend0_v'] = {'sec':'dend0','loc':0.41267884,'var':'v','conds':{'pop':'ball_stick_pop','cellLabel':0}}
                                
        # Column: dend_v3: Pop: ball_stick_pop; cell: 0; segment id: 3; segment name: Seg2_dend0; Neuron loc: dend0(0.8042263); value: v (v)
        self.simConfig.recordTraces['output0_ball_stick_pop_0_Seg2_dend0_v'] = {'sec':'dend0','loc':0.8042263,'var':'v','conds':{'pop':'ball_stick_pop','cellLabel':0}}
                        
        
        self.simConfig.plotCells = ['all']

        self.simConfig.recordStim = True  # record spikes of cell stims
        self.simConfig.recordStep = self.simConfig.dt # Step size in ms to save data (eg. V traces, LFP, etc)

        # Analysis and plotting, see http://neurosimlab.org/netpyne/reference.html#analysis-related-functions
        self.simConfig.analysis['plotRaster'] = False  # Plot raster
        self.simConfig.analysis['plot2Dnet'] = False  # Plot 2D net cells and connections
        self.simConfig.analysis['plotSpikeHist'] = False # plot spike histogram
        self.simConfig.analysis['plotConn'] = False # plot network connectivity
        self.simConfig.analysis['plotSpikePSD'] = False # plot 3d architecture

        # Saving
        self.simConfig.filename = 'ball_stick_net.txt'  # Set file output name
        self.simConfig.saveFileStep = self.simConfig.dt # step size in ms to save data to disk
        # self.simConfig.saveDat = True # save to dat file
        self.simConfig.saveJson = save_json # save to json file


    def run(self):

        ###############################################################################
        # IMPORT & RUN
        ###############################################################################

        print("Running a NetPyNE based simulation for %sms (dt: %sms) at %s degC"%(self.simConfig.duration, self.simConfig.dt, self.simConfig.hParams['celsius']))

        self.setup_sim_start = time.time()
        self.gids = sim.importNeuroML2SimulateAnalyze(self.nml2_file_name,self.simConfig)

        self.sim_end = time.time()
        self.setup_sim_time = self.sim_end - self.setup_sim_start
        print("Finished NetPyNE simulation in %f seconds (%f mins)..."%(self.setup_sim_time, self.setup_sim_time/60.0))

        try:
            self.save_results()
        except Exception as e:
            print("Exception saving results of NetPyNE simulation: %s" % (e))
            return

    def generate_json_only(self):

          ###############################################################################
          # GENERATE NETPYNE JSON REPRESENTATION OF NETWORK
          ###############################################################################

          print("Generating NetPyNE JSON (and mod files)")

          self.simConfig.saveJson = True # save to json file
          from netpyne.conversion.neuromlFormat import importNeuroML2
          self.gids = sim.importNeuroML2(self.nml2_file_name,
                                         self.simConfig,
                                         simulate=False,
                                         analyze=False)

          from netpyne.sim.save import saveData

          json_filename=__file__.replace(".py","")
          saveData(filename=json_filename, include=["simConfig", "netParams", "net"])
          real_json_filename='%s_data.json'%json_filename

          print("Finished exporting the NetPyNE JSON to %s"%real_json_filename)

          return real_json_filename


    def save_results(self):

        ###############################################################################
        #   Saving data (this ensures the data gets saved in the format/files
        #   as specified in the LEMS <Simulation> element)
        ###############################################################################

        if sim.rank==0:
        
            print("Saving traces to file: v.netpyne.dat (ref: output0)")

                
            # Column: t
            col_output0_t = [i*self.simConfig.dt for i in range(int(self.simConfig.duration/self.simConfig.dt))]
                        
            # Column: soma_v: Pop: ball_stick_pop; cell: 0; segment id: 0; segment name: Seg0_soma0; value: v
            col_output0_soma_v = sim.allSimData['output0_ball_stick_pop_0_Seg0_soma0_v']['cell_%s'%self.gids['ball_stick_pop'][0]]
                        
            # Column: dend_v1: Pop: ball_stick_pop; cell: 0; segment id: 1; segment name: Seg0_dend0; value: v
            col_output0_dend_v1 = sim.allSimData['output0_ball_stick_pop_0_Seg0_dend0_v']['cell_%s'%self.gids['ball_stick_pop'][0]]
                        
            # Column: dend_v2: Pop: ball_stick_pop; cell: 0; segment id: 2; segment name: Seg1_dend0; value: v
            col_output0_dend_v2 = sim.allSimData['output0_ball_stick_pop_0_Seg1_dend0_v']['cell_%s'%self.gids['ball_stick_pop'][0]]
                        
            # Column: dend_v3: Pop: ball_stick_pop; cell: 0; segment id: 3; segment name: Seg2_dend0; value: v
            col_output0_dend_v3 = sim.allSimData['output0_ball_stick_pop_0_Seg2_dend0_v']['cell_%s'%self.gids['ball_stick_pop'][0]]
                
            dat_file_output0 = open('v.netpyne.dat', 'w')
            for i in range(len(col_output0_t)):
                dat_file_output0.write( '%s\t'%(col_output0_t[i]/1000.0) +  '%s\t'%(col_output0_soma_v[i]/1000.0) +  '%s\t'%(col_output0_dend_v1[i]/1000.0) +  '%s\t'%(col_output0_dend_v2[i]/1000.0) +  '%s\t'%(col_output0_dend_v3[i]/1000.0) +  '\n')
            dat_file_output0.close()

        
        
            save_end = time.time()
            save_time = save_end - self.sim_end
            print("Finished saving results in %f seconds"%(save_time))


        
if __name__ == '__main__':

    save_json = '-json' in sys.argv
    no_run = '-norun' in sys.argv

    ns = NetPyNESimulation(tstop=1000.0, dt=0.025, seed=12345, save_json=save_json)

    if not no_run:
      ns.run()
    else:
      if save_json:
        fn = ns.generate_json_only()
        print("Generated: %s"%fn)
        quit()

    if '-nogui' in sys.argv:
        quit()
