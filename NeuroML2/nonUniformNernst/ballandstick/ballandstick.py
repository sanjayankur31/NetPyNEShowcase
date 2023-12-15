#!/usr/bin/env python3
"""
Simple ball and stick cell

File: NeuroML2/nonUniformNernst/ballandstick/ballandstick.py

Copyright 2023 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import shutil
import glob
import os
import math
import neuroml
import numpy
from neuroml.utils import component_factory
from pyneuroml.pynml import (
    write_neuroml2_file,
    run_lems_with_jneuroml_neuron,
    run_lems_with_jneuroml_netpyne,
)
from pyneuroml.lems import LEMSSimulation
from pyneuroml.plot import generate_plot


radius = math.sqrt(1000 / (4 * math.pi))

doc = component_factory(neuroml.NeuroMLDocument, id="ball_and_stick")

# create a simple ball and stick morphology
acell = component_factory(
    neuroml.Cell, id="acell", validate=False
)  # type: neuroml.Cell
acell.setup_nml_cell()
acell.info(True)
soma0 = acell.add_unbranched_segments(
    [(0, 0, 0, 2 * radius), (0, 0, 0, 2 * radius)], seg_type="soma", group_id="soma0"
)
acell.morphinfo(True)
dend0 = acell.add_unbranched_segments(
    [(radius, 0, 0, 10), (40, 0, 0, 1)],
    seg_type="dendrite",
    group_id="dend0",
    parent=acell.get_segment(0),
)
acell.morphinfo(True)

# biophysics
acell.set_spike_thresh("0 mV")
acell.set_resistivity("0.1 kohm_cm")
acell.set_specific_capacitance("1.0 uF_per_cm2", group_id="soma_group")
acell.set_specific_capacitance("2.0 uF_per_cm2", group_id="dendrite_group")
acell.set_init_memb_potential("-80 mV")

acell.add_channel_density(
    doc,
    "Ih_all",
    "Ih",
    "0.2 mS_per_cm2",
    "-45.0 mV",
    ion="hcn",
    ion_chan_def_file="./Ih.channel.nml",
)
acell.add_channel_density(
    doc,
    "K_Pst_soma",
    "K_Pst",
    "2.23 mS_per_cm2",
    "-85.0 mV",
    ion="k",
    ion_chan_def_file="./K_Pst.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "K_Tst_soma",
    "K_Tst",
    "81.2 mS_per_cm2",
    "-85.0 mV",
    ion="k",
    ion_chan_def_file="./K_Tst.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "Nap_Et2_soma",
    "Nap_Et2",
    "1.72 mS_per_cm2",
    "50.0 mV",
    ion="na",
    ion_chan_def_file="./Nap_Et2.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "NaTa_t_soma",
    "NaTa_t",
    "2040 mS_per_cm2",
    "50.0 mV",
    ion="na",
    ion_chan_def_file="./NaTa_t.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "SKv3_1_soma",
    "SKv3_1",
    "693.0 mS_per_cm2",
    "-85.0 mV",
    ion="k",
    ion_chan_def_file="./SKv3_1.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "SK_E2_soma",
    "SK_E2",
    "44.0 mS_per_cm2",
    "-85.0 mV",
    ion="k",
    ion_chan_def_file="./SK_E2.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "pas_soma",
    "pas",
    "0.0338 mS_per_cm2",
    "-90.0 mV",
    ion="non_specific",
    ion_chan_def_file="./pas.channel.nml",
    group_id="soma_group",
)
acell.add_channel_density(
    doc,
    "pas_dend",
    "pas",
    "0.0467 mS_per_cm2",
    "-90.0 mV",
    ion="non_specific",
    ion_chan_def_file="./pas.channel.nml",
    group_id="dendrite_group",
)

acell.biophysinfo()
doc.add(acell)

# network
network = doc.add(neuroml.Network, id="ball_stick_net", validate=False)
pop = network.add(neuroml.Population, id="ball_stick_pop", size="1",
                  component=acell.id, type="populationList", validate=False)
pop.add(neuroml.Instance, id=0, location=component_factory(neuroml.Location,
                                                           x=0, y=0, z=0))

# inputs
pg = doc.add(
    neuroml.PulseGenerator,
    id="pg0",
    delay="250ms",
    duration="500ms",
    amplitude="40pA",
)
network.add(neuroml.ExplicitInput, target=f"{pop.id}[0]", input=pg.id,
            destination="synapses")

# write to file
doc.validate(True)
write_neuroml2_file(doc, "ballandstick.net.nml")


# neuron sim
newsim = LEMSSimulation(
    "ballandstick_neuron", duration=1000, dt=0.025, target=network.id
)
newsim.include_neuroml2_file("ballandstick.net.nml")
newsim.create_output_file("output0", "v.neuron.dat")
newsim.add_column_to_output_file("output0", "soma_v",
                                 "ball_stick_pop/0/acell/0/v")
newsim.add_column_to_output_file("output0", "dend_v",
                                 "ball_stick_pop/0/acell/1/v")
simfile = newsim.save_to_file()
run_lems_with_jneuroml_neuron(simfile, nogui=True)
data_neuron = numpy.loadtxt("v.neuron.dat")

# clean up
shutil.rmtree("x86_64")
for f in glob.glob("*.mod"):
    os.remove(f)
for f in glob.glob("*.hoc"):
    os.remove(f)

# netpyne
newsim = LEMSSimulation(
    "ballandstick_netpyne", duration=1000, dt=0.025, target=network.id
)
newsim.include_neuroml2_file("ballandstick.net.nml")
newsim.create_output_file("output0", "v.netpyne.dat")
newsim.add_column_to_output_file("output0", "soma_v",
                                 "ball_stick_pop/0/acell/0/v")
newsim.add_column_to_output_file("output0", "dend_v",
                                 "ball_stick_pop/0/acell/1/v")
simfile = newsim.save_to_file()


run_lems_with_jneuroml_netpyne(simfile, nogui=True)
data_netpyne = numpy.loadtxt("v.netpyne.dat")

# clean up
shutil.rmtree("x86_64")
for f in glob.glob("*.mod"):
    os.remove(f)
for f in glob.glob("*.hoc"):
    os.remove(f)

generate_plot(
    [data_neuron[:, 0], data_netpyne[:, 0]],
    [data_neuron[:, 1], data_netpyne[:, 1]],
    title="Memb pot soma",
    labels=["nrn", "netpyne"],
    xaxis="time (s)",
    yaxis="v (mV)"
)

generate_plot(
    [data_neuron[:, 0], data_netpyne[:, 0]],
    [data_neuron[:, 2], data_netpyne[:, 2]],
    title="Memb pot dend",
    labels=["nrn", "netpyne"],
    xaxis="time (s)",
    yaxis="v (mV)"
)
