#!/usr/bin/env python3
"""
Simple ball and stick cell

File: NeuroML2/nonUniformNernst/ballandstick/ballandstick.py

Copyright 2023 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import glob
import math
import os
import shutil

import neuroml
import numpy
from neuroml.utils import component_factory
from pyneuroml.lems import LEMSSimulation
from pyneuroml.plot import generate_plot
from pyneuroml.pynml import (
    run_lems_with,
    write_neuroml2_file,
)


def clean(data=False):
    """Clean up"""
    # clean up
    shutil.rmtree("x86_64", ignore_errors=True)
    for f in glob.glob("*.mod"):
        os.remove(f)
    for f in glob.glob("*.hoc"):
        os.remove(f)
    if data is True:
        for f in glob.glob("*.dat"):
            os.remove(f)


def create():
    """Create model and sim"""
    radius = math.sqrt(1000 / (4 * math.pi))

    doc = component_factory(neuroml.NeuroMLDocument, id="ball_and_stick")

    # create a simple ball and stick morphology
    acell = component_factory(
        neuroml.Cell, id="acell", validate=False
    )  # type: neuroml.Cell
    acell.setup_nml_cell()
    acell.info(True)
    soma0 = acell.add_unbranched_segments(
        [(0, 0, 0, 2 * radius), (0, 0, 0, 2 * radius)],
        seg_type="soma",
        group_id="soma0",
    )
    acell.morphinfo(True)
    dend0 = acell.add_unbranched_segments(
        [
            (radius, 0, 0, radius * 1.0),
            (20, 0, 0, radius * 0.8),
            (30, 0, 0, radius * 0.5),
            (40, 0, 0, radius * 0.4),
        ],
        seg_type="dendrite",
        group_id="dend0",
        parent=acell.get_segment(0),
    )

    dendrite_group = acell.get_segment_group("dendrite_group")

    inhom = dendrite_group.add(
        neuroml.InhomogeneousParameter,
        id="PathLengthOverDend",
        variable="p",
        metric="Path Length from root",
        validate=False,
    )
    inhom.add(neuroml.ProximalDetails, translation_start="0")

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

    # non uniform nernst
    ca_lva = acell.add_channel_density_v(
        neuroml.ChannelDensityNonUniformNernst,
        doc,
        ion_chan_def_file="Ca_LVAst.channel.nml",
        id="Ca_LVAst_dend",
        ion="ca",
        ion_channel="Ca_LVAst",
        validate=False,
    )
    var_parm = ca_lva.add(
        neuroml.VariableParameter,
        parameter="condDensity",
        segment_groups="dendrite_group",
    )

    var_parm.add(
        neuroml.InhomogeneousValue,
        inhomogeneous_parameters="PathLengthOverDend",
        value="(1.0E-1 * p)",
    )

    acell.biophysinfo()
    doc.add(acell)

    # network
    network = doc.add(neuroml.Network, id="ball_stick_net", validate=False)
    pop = network.add(
        neuroml.Population,
        id="ball_stick_pop",
        size="1",
        component=acell.id,
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
        delay="250ms",
        duration="500ms",
        amplitude="45pA",
    )
    network.add(
        neuroml.ExplicitInput,
        target=f"{pop.id}[0]",
        input=pg.id,
        destination="synapses",
    )

    # write to file
    doc.validate(True)
    write_neuroml2_file(doc, "ballandstick.net.nml")

    data = {}
    # sims for neuron and netpyne engines
    for eng in ["neuron", "netpyne"]:
        # neuron sim
        newsim = LEMSSimulation(
            f"ballandstick_{eng}", duration=1000, dt=0.025, target=network.id
        )
        newsim.include_lems_file("Ca_LVAst.channel.nml")
        newsim.include_neuroml2_file("ballandstick.net.nml")
        newsim.create_output_file("output0", f"v.{eng}.dat")
        newsim.add_column_to_output_file(
            "output0", "soma_v", "ball_stick_pop/0/acell/0/v"
        )
        newsim.add_column_to_output_file(
            "output0", "dend_v1", "ball_stick_pop/0/acell/1/v"
        )
        newsim.add_column_to_output_file(
            "output0", "dend_v2", "ball_stick_pop/0/acell/2/v"
        )
        newsim.add_column_to_output_file(
            "output0", "dend_v3", "ball_stick_pop/0/acell/3/v"
        )
        simfile = newsim.save_to_file()
        run_lems_with(f"jneuroml_{eng}", simfile, nogui=True)
        data[eng] = numpy.loadtxt(f"v.{eng}.dat")
        clean()

    data_neuron = data["neuron"]
    data_netpyne = data["netpyne"]
    for x in range(0, 4):
        generate_plot(
            [data_neuron[:, 0], data_netpyne[:, 0]],
            [data_neuron[:, x + 1], data_netpyne[:, x + 1]],
            title=f"Memb pot seg {x}",
            labels=["nrn", "netpyne"],
            xaxis="time (s)",
            yaxis="v (mV)",
        )


if __name__ == "__main__":
    clean()
    create()
    # clean()
