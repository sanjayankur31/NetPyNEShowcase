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
import itertools
import statistics

import neuroml
import numpy
from neuroml.utils import component_factory
from pyneuroml.lems import LEMSSimulation
from pyneuroml.plot import generate_plot
from pyneuroml.plot.PlotMorphologyVispy import (plot_3D_cell_morphology,
                                                plot_interactive_3D)
from pyneuroml.utils.components import add_new_component
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
    dend0 = acell.add_unbranched_segments(
        [
            (radius, 0, 0, radius * 1.0),
            (20, 0, 0, radius * 0.8),
            (30, 0, 0, radius * 0.5),
            (40, 0, 0, radius * 0.4),
            (50, 0, 0, radius * 0.4),
            (60, 0, 0, radius * 0.4),
            (70, 0, 0, radius * 0.4),
            (80, 0, 0, radius * 0.4),
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

    acell.add_intracellular_property(
        "Species", id="ca", ion="ca", concentration_model="CaClamp",
        initial_concentration="5E-5 mM",
        initial_ext_concentration="2.0 mM"
    )

    doc.add(neuroml.IncludeType, href="channels/CaClamp.nml")
    caclamp, caclamp_file = add_new_component(
        component_id="CaClamp", component_type="caClamp",
        conc0="5E-5 mM",
        conc1="5E-5mM", delay="500.0ms", duration="500.0ms",
        ion="ca"
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

    # highlight_spec = {
    #     "1": {"marker_size": ["10", "10"], "marker_type": "cylinder",
    #           "marker_color": "yellow"},
    #     "3": {"marker_size": ["10", "10"], "marker_type": "cylinder",
    #           "marker_color": "orange"}
    #     }

    # plot_3D_cell_morphology(cell=acell,
    #                         nogui=False,
    #                         highlight_spec=highlight_spec, plot_type="detailed")
    #
    # plot_interactive_3D(nml_file=acell,
    #                     nogui=False,
    #                     highlight_spec={"acell": highlight_spec}, plot_type="detailed")
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
    engines = ["neuron", "netpyne"]
    for eng in engines:
        # neuron sim
        newsim = LEMSSimulation(
            f"ballandstick_{eng}", duration=1000, dt=0.025, target=network.id
        )
        newsim.include_lems_file("Ca_LVAst.channel.nml")
        newsim.include_lems_file(caclamp_file)
        newsim.include_neuroml2_file("ballandstick.net.nml")
        newsim.create_output_file("output0", f"v.{eng}.dat")
        newsim.add_column_to_output_file(
            "output0", "soma_v", "ball_stick_pop/0/acell/0/v"
        )
        # record from every dendritic segment
        for sid in dend0.members:
            newsim.add_column_to_output_file(
                "output0", f"dend_v{sid.segments}",
                f"ball_stick_pop/0/acell/{sid.segments}/v"
            )
        simfile = newsim.save_to_file()
        run_lems_with(f"jneuroml_{eng}", simfile, nogui=True)
        data[eng] = numpy.loadtxt(f"v.{eng}.dat")

    for x in range(0, len(dend0.members)):
        xs = []
        ys = []
        for eng in engines:
            xs.append(data[eng][:, 0])
            ys.append(data[eng][:, x + 1])
        generate_plot(
            xs,
            ys,
            title=f"Memb pot seg {x}",
            labels=engines,
            xaxis="time (s)",
            yaxis="v (mV)",
        )
        engine_combinations = itertools.combinations(engines, 2)
        for e1, e2 in engine_combinations:
            data1 = data[e1][:, x + 1]
            data2 = data[e2][:, x + 1]
            diff = [v1 - v2 for v1, v2 in zip(data1, data2)]
            datalen = len(diff)
            print(f"Stats (diff): mean: {statistics.mean(diff)}, stdev: {statistics.pstdev(diff)}")
            print(f"Correlation: {statistics.correlation(data1[:datalen], data2[:datalen])}")


if __name__ == "__main__":
    clean()
    create()
