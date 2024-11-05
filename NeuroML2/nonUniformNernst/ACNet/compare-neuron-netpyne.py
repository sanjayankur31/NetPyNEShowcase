#!/usr/bin/env python3
"""
Script to generate a simple NeuroML model and run it in NEURON and NetPyNE to
compare results.

This can probably be generalised, but since different cells may have different
channels, it's easier to copy this and modify it as required for each cell.

File: compare-neuron-netpyne.py

Copyright 2024 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import sys
import neuroml
from neuroml.utils import component_factory
from pyneuroml.io import read_neuroml2_file, write_neuroml2_file
from pyneuroml.lems.LEMSSimulation import LEMSSimulation
from pyneuroml.runners import run_lems_with
from pyneuroml.plot import generate_plot
import numpy


celldoc: neuroml.NeuroMLDocument = read_neuroml2_file("pyr_4_sym.cell.nml")
acell: neuroml.Cell = celldoc.cells[0]

# add inhom param
all_segment_group = acell.get_segment_group("all")
inhom_param = all_segment_group.add(neuroml.InhomogeneousParameter,
                                    id="PathOverAllSegs", variable="p",
                                    metric="Path Length from root")

inhom_param.add(neuroml.ProximalDetails, translation_start="0")

# only keep leak
membrane_props = acell.biophysical_properties.membrane_properties

channel_densities = membrane_props.channel_densities
newcds = []

for cd in channel_densities:
    if "Leak" in cd.id:
        newcds.append(cd)
membrane_props.channel_densities = newcds
new_non_uniform_cd = membrane_props.add(neuroml.ChannelDensityNonUniform, id="Ih_all",
                                        ion_channel="Ih", ion="hcn", erev="-45mV", validate=False)

var_par = new_non_uniform_cd.add(neuroml.VariableParameter, parameter="condDensity",
                                 segment_groups="all", validate=False)

var_par.add(neuroml.InhomogeneousValue,
            inhomogeneous_parameters="PathOverAllSegs", value="0.0")

doc = component_factory(neuroml.NeuroMLDocument, id="pyr_4_sym_test")
doc.add(acell)
doc.includes.extend(celldoc.includes)

# network
network = doc.add(neuroml.Network, id="ac_net", validate=False)
pop = network.add(
    neuroml.Population,
    id="ac_pop",
    size="1",
    component="pyr_4_sym",
    type="populationList",
    validate=False,
)
pop.add(
    neuroml.Instance,
    id=0,
    location=component_factory(neuroml.Location, x=0, y=0, z=0),
)

# inputs
pg = doc.add(
    neuroml.PulseGenerator,
    id="pg0",
    delay="500ms",
    duration="500ms",
    amplitude="7.93E-10A",
)
input_list = network.add(
    neuroml.InputList,
    id="input_list_0",
    populations=f"{pop.id}",
    component=pg.id,
    validate=False
)

# input to the first dendritic segment, not soma
input_list.add(
    neuroml.Input,
    id="0",
    target=f"../{pop.id}/0/acell",
    destination="synapses",
    segment_id="0"
)

# write to file
doc.validate(True)
write_neuroml2_file(doc, "ac.net.nml")

data = {}
# sims for neuron and netpyne engines
engines = ["neuron", "netpyne"]
segs_record = [0, 8, 5, 4]
for eng in engines:
    # neuron sim
    newsim = LEMSSimulation(
        f"ac_{eng}", duration=1500, dt=0.025, target=network.id
    )

    newsim.include_lems_file("./Ca_conc.nml")
    newsim.include_lems_file("./Ca_pyr.channel.nml")
    newsim.include_lems_file("./Kahp_pyr.channel.nml")
    newsim.include_lems_file("./Kdr_pyr.channel.nml")
    newsim.include_lems_file("./LeakConductance_pyr.channel.nml")
    newsim.include_lems_file("./Na_pyr.channel.nml")
    newsim.include_lems_file("./Ih.channel.nml")

    newsim.include_neuroml2_file("ac.net.nml")
    newsim.create_output_file("output0", f"v.{eng}.dat")

    # record from every dendritic segment
    for sid in segs_record:
        newsim.add_column_to_output_file(
            "output0", f"dend_v{sid}",
            f"ac_pop/0/pyr_4_sym/{sid}/v"
        )
    simfile = newsim.save_to_file()
    run_lems_with(f"jneuroml_{eng}", simfile, nogui=True)
    data[eng] = numpy.loadtxt(f"v.{eng}.dat")

xs = []
ys = []
labels = []
for x in range(0, len(segs_record)):
    for eng in engines:
        xs.append(data[eng][:, 0])
        ys.append(data[eng][:, x + 1])
        labels.append(f"dend_{segs_record[x]}_{eng}")

generate_plot(
    xs,
    ys,
    title="v",
    labels=labels,
    xaxis="time (s)",
    yaxis="v (mV)",
)
