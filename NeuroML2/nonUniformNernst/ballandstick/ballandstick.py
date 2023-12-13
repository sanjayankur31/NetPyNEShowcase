#!/usr/bin/env python3
"""
Simple ball and stick cell

File: NeuroML2/nonUniformNernst/ballandstick/ballandstick.py

Copyright 2023 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import math
import neuroml
from neuroml.utils import component_factory
from pyneuroml.pynml import write_neuroml2_file


radius = math.sqrt(1000/(4*math.pi))

doc = component_factory(neuroml.NeuroMLDocument, id="ball_and_stick")

# create a simple ball and stick morphology
acell = component_factory(neuroml.Cell, id="acell", validate=False)  # type: neuroml.Cell
acell.setup_nml_cell()
acell.info(True)
soma_0 = acell.add_unbranched_segments([
    (0, 0, 0, 2*radius), (0, 0, 0, 2*radius)], seg_type="soma", group_id="soma_0"
                                       )
acell.morphinfo(True)
acell.add_unbranched_segments([
    (radius, 0, 0, 10), (40, 0, 0, 5)], seg_type="dendrite", group_id="dend_0",
                              parent=acell.get_segment(0),
)
acell.morphinfo(True)
doc.add(acell)


# write to file
doc.validate(True)
write_neuroml2_file(doc, "ballandstick.cell.nml")
