TITLE Mod file for component: Component(id=Kdr_bask type=ionChannelHH)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.10.1
         org.neuroml.model   v1.10.1
         jLEMS               v0.11.1

ENDCOMMENT

NEURON {
    SUFFIX Kdr_bask
    USEION k WRITE ik VALENCE 1 ? Assuming valence = 1; TODO check this!!
    
    RANGE gion
    RANGE i__Kdr_bask : a copy of the variable for current which makes it easier to access from outside the mod file
    RANGE gmax                              : Will be changed when ion channel mechanism placed on cell!
    RANGE conductance                       : parameter
    RANGE g                                 : exposure
    RANGE fopen                             : exposure
    RANGE n_instances                       : parameter
    RANGE n_alpha                           : exposure
    RANGE n_beta                            : exposure
    RANGE n_tau                             : exposure
    RANGE n_inf                             : exposure
    RANGE n_rateScale                       : exposure
    RANGE n_fcond                           : exposure
    RANGE n_forwardRate_rate                : parameter
    RANGE n_forwardRate_midpoint            : parameter
    RANGE n_forwardRate_scale               : parameter
    RANGE n_forwardRate_r                   : exposure
    RANGE n_reverseRate_rate                : parameter
    RANGE n_reverseRate_midpoint            : parameter
    RANGE n_reverseRate_scale               : parameter
    RANGE n_reverseRate_r                   : exposure
    RANGE n_forwardRate_x                   : derived variable
    RANGE conductanceScale                  : derived variable
    RANGE fopen0                            : derived variable
    
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
    
    gmax = 0  (S/cm2)                       : Will be changed when ion channel mechanism placed on cell!
    
    conductance = 1.0E-5 (uS)              : was: 1.0E-11 (conductance)
    n_instances = 4                        : was: 4.0 (none)
    n_forwardRate_rate = 0.32000002 (kHz)  : was: 320.0 (per_time)
    n_forwardRate_midpoint = -48 (mV)      : was: -0.048 (voltage)
    n_forwardRate_scale = 5 (mV)           : was: 0.005 (voltage)
    n_reverseRate_rate = 1 (kHz)           : was: 1000.0 (per_time)
    n_reverseRate_midpoint = -53 (mV)      : was: -0.053 (voltage)
    n_reverseRate_scale = -40 (mV)         : was: -0.04 (voltage)
}

ASSIGNED {
    
    gion   (S/cm2)                          : Transient conductance density of the channel? Standard Assigned variables with ionChannel
    v (mV)
    celsius (degC)
    temperature (K)
    ek (mV)
    ik (mA/cm2)
    i__Kdr_bask (mA/cm2)
    
    n_forwardRate_x                         : derived variable
    
    n_forwardRate_r (kHz)                   : conditional derived var...
    n_reverseRate_r (kHz)                   : derived variable
    n_rateScale                             : derived variable
    n_alpha (kHz)                           : derived variable
    n_beta (kHz)                            : derived variable
    n_fcond                                 : derived variable
    n_inf                                   : derived variable
    n_tau (ms)                              : derived variable
    conductanceScale                        : derived variable
    fopen0                                  : derived variable
    fopen                                   : derived variable
    g (uS)                                  : derived variable
    rate_n_q (/ms)
    
}

STATE {
    n_q  : dimension: none
    
}

INITIAL {
    ek = -90.0
    
    temperature = celsius + 273.15
    
    rates()
    rates() ? To ensure correct initialisation.
    
    n_q = n_inf
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    ? DerivedVariable is based on path: conductanceScaling[*]/factor, on: Component(id=Kdr_bask type=ionChannelHH), from conductanceScaling; null
    ? Path not present in component, using factor: 1
    
    conductanceScale = 1 
    
    ? DerivedVariable is based on path: gates[*]/fcond, on: Component(id=Kdr_bask type=ionChannelHH), from gates; Component(id=n type=gateHHrates)
    ? multiply applied to all instances of fcond in: <gates> ([Component(id=n type=gateHHrates)]))
    fopen0 = n_fcond ? path based, prefix = 
    
    fopen = conductanceScale  *  fopen0 ? evaluable
    g = conductance  *  fopen ? evaluable
    gion = gmax * fopen 
    
    ik = gion * (v - ek)
    i__Kdr_bask =  -1 * ik : set this variable to the current also - note -1 as channel current convention for LEMS used!
    
}

DERIVATIVE states {
    rates()
    n_q' = rate_n_q 
    
}

PROCEDURE rates() {
    
    n_forwardRate_x = (v -  n_forwardRate_midpoint ) /  n_forwardRate_scale ? evaluable
    if (n_forwardRate_x  != 0)  { 
        n_forwardRate_r = n_forwardRate_rate  *  n_forwardRate_x  / (1 - exp(0 -  n_forwardRate_x )) ? evaluable cdv
    } else if (n_forwardRate_x  == 0)  { 
        n_forwardRate_r = n_forwardRate_rate ? evaluable cdv
    }
    
    n_reverseRate_r = n_reverseRate_rate  * exp((v -  n_reverseRate_midpoint )/ n_reverseRate_scale ) ? evaluable
    ? DerivedVariable is based on path: q10Settings[*]/q10, on: Component(id=n type=gateHHrates), from q10Settings; null
    ? Path not present in component, using factor: 1
    
    n_rateScale = 1 
    
    ? DerivedVariable is based on path: forwardRate/r, on: Component(id=n type=gateHHrates), from forwardRate; Component(id=null type=HHExpLinearRate)
    n_alpha = n_forwardRate_r ? path based, prefix = n_
    
    ? DerivedVariable is based on path: reverseRate/r, on: Component(id=n type=gateHHrates), from reverseRate; Component(id=null type=HHExpRate)
    n_beta = n_reverseRate_r ? path based, prefix = n_
    
    n_fcond = n_q ^ n_instances ? evaluable
    n_inf = n_alpha /( n_alpha + n_beta ) ? evaluable
    n_tau = 1/(( n_alpha + n_beta ) *  n_rateScale ) ? evaluable
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    rate_n_q = ( n_inf  -  n_q ) /  n_tau ? Note units of all quantities used here need to be consistent!
    
     
    
     
    
     
    
}

