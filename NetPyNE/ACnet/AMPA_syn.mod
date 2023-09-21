TITLE Mod file for component: Component(id=AMPA_syn type=expTwoSynapse)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.9.1
         org.neuroml.model   v1.9.1
         jLEMS               v0.10.8

ENDCOMMENT

NEURON {
    POINT_PROCESS AMPA_syn
    RANGE tauRise                           : parameter
    RANGE tauDecay                          : parameter
    RANGE peakTime                          : parameter
    RANGE waveformFactor                    : parameter
    RANGE gbase                             : parameter
    RANGE erev                              : parameter
    RANGE g                                 : exposure
    RANGE i                                 : exposure
    
    
    NONSPECIFIC_CURRENT i 
    
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
    
    tauRise = 3 (ms)                       : was: 0.003 (time)
    tauDecay = 3.1 (ms)                    : was: 0.0031 (time)
    peakTime = 3.0494535 (ms)              : was: 0.0030494535225381453 (time)
    waveformFactor = 82.903885             : was: 82.9038820519792 (none)
    gbase = 0.03 (uS)                      : was: 3.0E-8 (conductance)
    erev = 0 (mV)                          : was: 0.0 (voltage)
}

ASSIGNED {
    ? Standard Assigned variables with baseSynapse
    v (mV)
    celsius (degC)
    temperature (K)
    g (uS)                                  : derived variable
    i (nA)                                  : derived variable
    rate_A (/ms)
    rate_B (/ms)
    
}

STATE {
    A  : dimension: none
    B  : dimension: none
    
}

INITIAL {
    temperature = celsius + 273.15
    
    rates()
    rates() ? To ensure correct initialisation.
    
    A = 0
    
    B = 0
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    
}

NET_RECEIVE(weight) {
    
    : paramMappings . : {AMPA_syn={A=A, B=B, tauRise=tauRise, tauDecay=tauDecay, peakTime=peakTime, waveformFactor=waveformFactor, gbase=gbase, erev=erev, g=g, i=i}, AMPA_syn_notes={}}
    : state_discontinuity(A, A  + (weight *   waveformFactor  )) : From AMPA_syn
    A = A  + (weight *   waveformFactor  ) : From AMPA_syn
    
    : paramMappings . : {AMPA_syn={A=A, B=B, tauRise=tauRise, tauDecay=tauDecay, peakTime=peakTime, waveformFactor=waveformFactor, gbase=gbase, erev=erev, g=g, i=i}, AMPA_syn_notes={}}
    : state_discontinuity(B, B  + (weight *   waveformFactor  )) : From AMPA_syn
    B = B  + (weight *   waveformFactor  ) : From AMPA_syn
    
}

DERIVATIVE states {
    rates()
    A' = rate_A 
    B' = rate_B 
    
}

PROCEDURE rates() {
    
    g = gbase  * (  B   -   A  ) ? evaluable
    i = -1 * g  * (  erev   - v) ? evaluable
    rate_A = -  A   /  tauRise ? Note units of all quantities used here need to be consistent!
    rate_B = -  B   /  tauDecay ? Note units of all quantities used here need to be consistent!
    
     
    
}

