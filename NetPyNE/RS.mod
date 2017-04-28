TITLE Mod file for component: Component(id=RS type=izhikevich2007Cell)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.5.0
         org.neuroml.model   v1.5.0
         jLEMS               v0.9.8.7

ENDCOMMENT

NEURON {
    POINT_PROCESS RS
    
    
    NONSPECIFIC_CURRENT i                    : To ensure v of section follows vI
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
    
    RANGE copy_v                           : copy of v on section
    
}

UNITS {
    
    (nA) = (nanoamp)
    (uA) = (microamp)
    (mA) = (milliamp)
    (A) = (amp)
    (mV) = (millivolt)
    (mS) = (millisiemens)
    (uS) = (microsiemens)
    (molar) = (1/liter)
    (kHz) = (kilohertz)
    (mM) = (millimolar)
    (um) = (micrometer)
    (umol) = (micromole)
    (S) = (siemens)
    
}

PARAMETER {
    
    v0 = -60 (mV)
    k = 7.0E-4 (uS / mV)
    vr = -60 (mV)
    vt = -40 (mV)
    vpeak = 35 (mV)
    a = 0.030000001 (kHz)
    b = -0.0019999999 (uS)
    c = -50 (mV)
    d = 0.1 (nA)
    C = 1.00000005E-4 (microfarads)
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
    
    copy_v (mV)
    
    
    iSyn (nA)                              : derived variable
    
    iMemb (nA)                             : derived variable
    rate_v (mV/ms)
    rate_u (nA/ms)
    
}

STATE {
    vI (nA) 
    u (nA) 
    
}

INITIAL {
    rates()
    rates() ? To ensure correct initialisation.
    
    net_send(0, 1) : go to NET_RECEIVE block, flag 1, for initial state
    
    u = 0
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    
    copy_v = v
    i = vI * C
}

NET_RECEIVE(flag) {
    
    if (flag == 1) { : Setting watch for top level OnCondition...
        WATCH (v >  vpeak) 1000
    }
    if (flag == 1000) {
    
        v = c
    
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
    
    vI = -1 * rate_v
     
    
}

