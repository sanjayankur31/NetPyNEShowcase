'''
NETPYNE simulator compliant export for:

Components:
    gj1 (Type: gapJunction:  conductance=1.0E-11 (SI conductance))
    iaf (Type: iafCell:  leakConductance=2.0000000000000003E-10 (SI conductance) leakReversal=-0.07 (SI voltage) thresh=-0.055 (SI voltage) reset=-0.07 (SI voltage) C=3.2E-12 (SI capacitance))
    pulseGen1 (Type: pulseGenerator:  delay=0.05 (SI time) duration=0.2 (SI time) amplitude=3.2000000000000005E-12 (SI current))
    pulseGen2 (Type: pulseGenerator:  delay=0.4 (SI time) duration=0.2 (SI time) amplitude=3.2000000000000005E-12 (SI current))
    net1 (Type: network)
    sim1 (Type: Simulation:  length=0.7000000000000001 (SI time) step=1.0E-5 (SI time))


    This NETPYNE file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.9.0
         org.neuroml.model   v1.9.0
         jLEMS               v0.10.7

'''
# Main NetPyNE script for: net1

# See https://github.com/Neurosim-lab/netpyne

from netpyne import specs  # import netpyne specs module
from netpyne import sim    # import netpyne sim module
from netpyne import __version__ as version

from neuron import h

import sys
import time
import datetime

class NetPyNESimulation():

    def __init__(self, tstop=700.0, dt=0.01, seed=123456789, save_json=False):

        self.setup_start = time.time()
        
        self.report_file = open('report.gj.txt','w')
        self.report_file.write('# Report of running simulation with NetPyNE v%s\n'%version)
        self.report_file.write('Simulator=NetPyNE\n')
        self.report_file.write('SimulatorVersion=%s\n'%version)
        self.report_file.write('SimulationFile=%s\n'%__file__)
        self.report_file.write('PythonVersion=%s\n'%sys.version.replace('\n',' '))
        self.report_file.write('NeuronVersion=%s\n'%h.nrnversion())
        self.report_file.write('NeuroMLExportVersion=1.9.0\n')
        self.report_file.close()
        

        ###############################################################################
        # NETWORK PARAMETERS
        ###############################################################################

        self.nml2_file_name = 'GJ.nml'

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
        self.simConfig.seeds = {'conn': 0, 'stim': 123456789, 'loc': 0}

        self.simConfig.createNEURONObj = 1  # create HOC objects when instantiating network
        self.simConfig.createPyStruct = 1  # create Python structure (simulator-independent) when instantiating network
        self.simConfig.verbose = False  # show detailed messages
        
        # Recording
        self.simConfig.recordCells = ['all']
        self.simConfig.recordTraces = {}
        self.simConfig.saveCellSecs=False
        self.simConfig.saveCellConns=False
        self.simConfig.gatherOnlySimData=True

                # For saving to file: ex19_v.dat (ref: of0)
                                
        # Column: iafCell1_0: Pop: iafPop1; cell: 0; segment id: $oc.segment_id; segment name: soma; Neuron loc: soma(0.5); value: v (v)
        self.simConfig.recordTraces['of0_iafPop1_0_soma_v'] = {'sec':'soma','loc':0.5,'var':'v','conds':{'pop':'iafPop1','cellLabel':0}}
                                
        # Column: iafCell2_0: Pop: iafPop2; cell: 0; segment id: $oc.segment_id; segment name: soma; Neuron loc: soma(0.5); value: v (v)
        self.simConfig.recordTraces['of0_iafPop2_0_soma_v'] = {'sec':'soma','loc':0.5,'var':'v','conds':{'pop':'iafPop2','cellLabel':0}}
                        
        
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
        self.simConfig.filename = 'net1.txt'  # Set file output name
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

          saveData(filename=self.nml2_file_name.replace(".nml",""), include=["simConfig", "netParams", "net"])

          probable_filenames = [self.nml2_file_name.replace(".nml","")+"_data.json"] # may change in netpyne core...
          print("Finished NetPyNE JSON")

          return probable_filenames


    def save_results(self):

        ###############################################################################
        #   Saving data (this ensures the data gets saved in the format/files
        #   as specified in the LEMS <Simulation> element)
        ###############################################################################

        if sim.rank==0:
        
            print("Saving traces to file: ex19_v.dat (ref: of0)")

                
            # Column: t
            col_of0_t = [i*self.simConfig.dt for i in range(int(self.simConfig.duration/self.simConfig.dt))]
                        
            # Column: iafCell1_0: Pop: iafPop1; cell: 0; segment id: $oc.segment_id; segment name: soma; value: v
            col_of0_iafCell1_0 = sim.allSimData['of0_iafPop1_0_soma_v']['cell_%s'%self.gids['iafPop1'][0]]
                        
            # Column: iafCell2_0: Pop: iafPop2; cell: 0; segment id: $oc.segment_id; segment name: soma; value: v
            col_of0_iafCell2_0 = sim.allSimData['of0_iafPop2_0_soma_v']['cell_%s'%self.gids['iafPop2'][0]]
                
            dat_file_of0 = open('ex19_v.dat', 'w')
            for i in range(len(col_of0_t)):
                dat_file_of0.write( '%s\t'%(col_of0_t[i]/1000.0) +  '%s\t'%(col_of0_iafCell1_0[i]/1000.0) +  '%s\t'%(col_of0_iafCell2_0[i]/1000.0) +  '\n')
            dat_file_of0.close()

        
        
            save_end = time.time()
            save_time = save_end - self.sim_end
            print("Finished saving results in %f seconds"%(save_time))


        
            self.report_file = open('report.gj.txt','a')
            self.report_file.write('StartTime=%s\n'%datetime.datetime.fromtimestamp(self.setup_start).strftime('%Y-%m-%d %H:%M:%S'))
            self.report_file.write('RealSetupAndSimulationTime=%s\n'%self.setup_sim_time)
            self.report_file.write('SimulationSaveTime=%s\n'%save_time)
            self.report_file.close()
            print("Saving report of simulation to %s"%('report.gj.txt'))
        
if __name__ == '__main__':

    save_json = '-json' in sys.argv
    no_run = '-norun' in sys.argv

    ns = NetPyNESimulation(tstop=700.0, dt=0.01, seed=123456789, save_json=save_json)

    if not no_run:
      ns.run()
    else:
      if save_json:
        fn = ns.generate_json_only()
        print("Generated: %s"%fn)
        quit()

    if '-nogui' in sys.argv:
        quit()
