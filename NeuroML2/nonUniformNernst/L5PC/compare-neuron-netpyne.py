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


import neuroml
from neuroml.utils import component_factory
from pyneuroml.io import read_neuroml2_file, write_neuroml2_file
from pyneuroml.lems.LEMSSimulation import LEMSSimulation
from pyneuroml.runners import run_lems_with
from pyneuroml.plot import generate_plot
import numpy


celldoc: neuroml.NeuroMLDocument = read_neuroml2_file("L5PC.cell.frac.nml")
acell: neuroml.Cell = celldoc.cells[0]

# remove all other channel densities
membrane_props = acell.biophysical_properties.membrane_properties

for attr in membrane_props.info(return_format="list"):
    if "channel_densit" in attr:
        # if attr == "channel_density_non_uniform_nernsts":
        if "non_uniform" in attr or "nernst" in attr:
            setattr(membrane_props, attr, [])
        """

        # only keep 1
        else:
            non_uniform_nernsts = getattr(membrane_props, attr)
            setattr(membrane_props, attr, [non_uniform_nernsts[0]])
        """

doc = component_factory(neuroml.NeuroMLDocument, id="L5PC_test")
doc.add(acell)
doc.includes.extend(celldoc.includes)

# network
network = doc.add(neuroml.Network, id="l5pc_net", validate=False)
pop = network.add(
    neuroml.Population,
    id="l5pc_pop",
    size="1",
    component="L5PC",
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
write_neuroml2_file(doc, "l5pc.net.nml")

data = {}
# sims for neuron and netpyne engines
engines = ["neuron", "netpyne"]
segs_record = [0, 13, 1176, 1454, 2330]
for eng in engines:
    # neuron sim
    newsim = LEMSSimulation(
        f"l5pc_{eng}", duration=1500, dt=0.025, target=network.id
    )

    newsim.include_lems_file("CaDynamics_E2_NML2__decay122__gamma5_09Emin4.nml")
    newsim.include_lems_file("Ca_HVA.channel.nml")
    newsim.include_lems_file("Im.channel.nml")
    newsim.include_lems_file("NaTa_t.channel.nml")
    newsim.include_lems_file("SKv3_1.channel.nml")
    newsim.include_lems_file("CaDynamics_E2_NML2__decay460__gamma5_01Emin4.nml")
    newsim.include_lems_file("Ca_LVAst.channel.nml")
    newsim.include_lems_file("K_Pst.channel.nml")
    newsim.include_lems_file("pas.channel.nml")
    newsim.include_lems_file("CaDynamics_E2_NML2.nml")
    newsim.include_lems_file("Ih.channel.nml")
    newsim.include_lems_file("K_Tst.channel.nml")
    newsim.include_lems_file("Nap_Et2.channel.nml")
    newsim.include_lems_file("SK_E2.channel.nml")

    newsim.include_neuroml2_file("l5pc.net.nml")
    newsim.create_output_file("output0", f"v.{eng}.dat")

    # record from every dendritic segment
    for sid in segs_record:
        newsim.add_column_to_output_file(
            "output0", f"dend_v{sid}",
            f"l5pc_pop/0/L5PC/{sid}/v"
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
