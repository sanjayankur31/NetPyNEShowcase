TITLE Mod file for component: Component(id=BackgroundRandomIClamps type=pulseGenerator)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.10.1
         org.neuroml.model   v1.10.1
         jLEMS               v0.11.1

ENDCOMMENT

NEURON {
    POINT_PROCESS BackgroundRandomIClamps
    ELECTRODE_CURRENT i
    RANGE weight                            : property
    RANGE delay                             : parameter
    RANGE duration                          : parameter
    RANGE amplitude                         : parameter
    
}

UNITS {
    
    (nA) = (nanoamp)
    (uA) = (microamp)
    (mA) = (milliamp)
    (A) = (amp)
    (mV) = (millivolt)
    (mS) = (millisiemens)
    (uS) = (microsiemens)
    (nF) = (nanofarad)
    (molar) = (1/liter)
    (kHz) = (kilohertz)
    (mM) = (millimolar)
    (um) = (micrometer)
    (umol) = (micromole)
    (pC) = (picocoulomb)
    (S) = (siemens)
    
}

PARAMETER {
    
    weight = 1
    delay = 0 (ms)                         : was: 0.0 (time)
    duration = 100000 (ms)                 : was: 100.0 (time)
    amplitude = 0.6 (nA)                   : was: 6.0E-10 (current)
}

STATE {
    i (nA) : dimension: current
    
}

INITIAL {
    rates()
    rates() ? To ensure correct initialisation.
    
}

BREAKPOINT {
    
    rates()
    if (t <  delay) {
        i = 0 ? standard OnCondition
    }
    
    if (t >=  delay  && t <  duration  +  delay) {
        i = weight  *  amplitude ? standard OnCondition
    }
    
    if (t >=  duration  +  delay) {
        i = 0 ? standard OnCondition
    }
    
    
}

PROCEDURE rates() {
    
    
     
    
}

