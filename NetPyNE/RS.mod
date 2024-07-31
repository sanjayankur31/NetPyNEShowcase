TITLE Mod file for component: Component(id=RS type=izhikevich2007Cell)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.10.1
         org.neuroml.model   v1.10.1
         jLEMS               v0.11.1

ENDCOMMENT

NEURON {
    POINT_PROCESS RS
    
    
    NONSPECIFIC_CURRENT i                   : To ensure v of section follows v_I
    RANGE v0                                : parameter
    RANGE k                                 : parameter
    RANGE vr                                : parameter
    RANGE vt                                : parameter
    RANGE vpeak                             : parameter
    RANGE a                                 : parameter
    RANGE b                                 : parameter
    RANGE c                                 : parameter
    RANGE d                                 : parameter
    RANGE C                                 : parameter
    RANGE iSyn                              : exposure
    RANGE iMemb                             : exposure
    
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
    
    v0 = -60 (mV)                          : was: -0.06 (voltage)
    k = 7.0E-4 (uS / mV)                   : was: 7.0E-7 (conductance_per_voltage)
    vr = -60 (mV)                          : was: -0.06 (voltage)
    vt = -40 (mV)                          : was: -0.04 (voltage)
    vpeak = 35 (mV)                        : was: 0.035 (voltage)
    a = 0.030000001 (kHz)                  : was: 30.0 (per_time)
    b = -0.002 (uS)                        : was: -2.0E-9 (conductance)
    c = -50 (mV)                           : was: -0.05 (voltage)
    d = 0.1 (nA)                           : was: 1.0E-10 (current)
    C = 0.1 (nF)                           : was: 1.0E-10 (capacitance)
}

ASSIGNED {
    v (mV)
    i (nA)                                 : the point process current 
    
    v_I (mV/ms)                             : for rate of change of voltage 
    iSyn (nA)                               : derived variable
    iMemb (nA)                              : derived variable
    rate_v (mV/ms)
    rate_u (nA/ms)
    
}

STATE {
    u (nA) : dimension: current
    
}

INITIAL {
    rates()
    rates() ? To ensure correct initialisation.
    
    net_send(0, 1) : go to NET_RECEIVE block, flag 1, for initial state
    
    u = 0
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    
    i = v_I * C
}

NET_RECEIVE(flag) {
    
    if (flag == 1) { : Setting watch for top level OnCondition...
        WATCH (v >  vpeak) 1000
    }
    if (flag == 1000) {
    
        v = c
    
        v_I = 0 : Setting rate of change of v to 0
    
        u = u  +  d
    }
    if (flag == 1) { : Set initial states
    
        v = v0
    }
    
}

DERIVATIVE states {
    rates()
    u' = rate_u 
    
}

PROCEDURE rates() {
    
    ? DerivedVariable is based on path: synapses[*]/i, on: Component(id=RS type=izhikevich2007Cell), from synapses; null
    iSyn = 0 ? Was: synapses[*]_i but insertion of currents from external attachments not yet supported ? path based, prefix = 
    
    iMemb = k  * (v-  vr  ) * (v-  vt  ) +  iSyn  -  u ? evaluable
    rate_v = iMemb  /  C ? Note units of all quantities used here need to be consistent!
    rate_u = a  * (  b   * (v-  vr  ) -   u  ) ? Note units of all quantities used here need to be consistent!
    
    v_I = -1 * rate_v
     
    
}

