TITLE Mod file for component: Component(id=Kahp_pyr type=ionChannelHH)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.9.0
         org.neuroml.model   v1.9.0
         jLEMS               v0.10.7

ENDCOMMENT

NEURON {
    SUFFIX Kahp_pyr
    USEION ca READ cai,cao VALENCE 2
    USEION k WRITE ik VALENCE 1 ? Assuming valence = 1; TODO check this!!
    
    RANGE gion
    RANGE i__Kahp_pyr : a copy of the variable for current which makes it easier to access from outside the mod file
    RANGE gmax                              : Will be changed when ion channel mechanism placed on cell!
    RANGE conductance                       : parameter
    
    RANGE g                                 : exposure
    
    RANGE fopen                             : exposure
    RANGE z_instances                       : parameter
    
    RANGE z_alpha                           : exposure
    
    RANGE z_beta                            : exposure
    
    RANGE z_tau                             : exposure
    
    RANGE z_inf                             : exposure
    
    RANGE z_rateScale                       : exposure
    
    RANGE z_fcond                           : exposure
    RANGE z_forwardRate_TIME_SCALE          : parameter
    RANGE z_forwardRate_VOLT_SCALE          : parameter
    RANGE z_forwardRate_CONC_SCALE          : parameter
    
    RANGE z_forwardRate_r                   : exposure
    RANGE z_reverseRate_TIME_SCALE          : parameter
    RANGE z_reverseRate_VOLT_SCALE          : parameter
    RANGE z_reverseRate_CONC_SCALE          : parameter
    
    RANGE z_reverseRate_r                   : exposure
    RANGE z_forwardRate_V                   : derived variable
    RANGE z_forwardRate_ca_conc             : derived variable
    RANGE z_reverseRate_V                   : derived variable
    RANGE z_reverseRate_ca_conc             : derived variable
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
    (molar) = (1/liter)
    (kHz) = (kilohertz)
    (mM) = (millimolar)
    (um) = (micrometer)
    (umol) = (micromole)
    (S) = (siemens)
    
}

PARAMETER {
    
    gmax = 0  (S/cm2)                       : Will be changed when ion channel mechanism placed on cell!
    
    conductance = 1.0E-5 (uS)
    z_instances = 1 
    z_forwardRate_TIME_SCALE = 1000 (ms)
    z_forwardRate_VOLT_SCALE = 1000 (mV)
    z_forwardRate_CONC_SCALE = 1 (mM)
    z_reverseRate_TIME_SCALE = 1000 (ms)
    z_reverseRate_VOLT_SCALE = 1000 (mV)
    z_reverseRate_CONC_SCALE = 1 (mM)
}

ASSIGNED {
    
    gion   (S/cm2)                          : Transient conductance density of the channel? Standard Assigned variables with ionChannel
    v (mV)
    celsius (degC)
    temperature (K)
    ek (mV)
    ik (mA/cm2)
    i__Kahp_pyr (mA/cm2)
    
    cai (mM)
    
    cao (mM)
    
    
    z_forwardRate_V                        : derived variable
    
    z_forwardRate_ca_conc                  : derived variable
    
    z_forwardRate_r (kHz)                  : conditional derived var...
    
    z_reverseRate_V                        : derived variable
    
    z_reverseRate_ca_conc                  : derived variable
    
    z_reverseRate_r (kHz)                  : derived variable
    
    z_rateScale                            : derived variable
    
    z_alpha (kHz)                          : derived variable
    
    z_beta (kHz)                           : derived variable
    
    z_fcond                                : derived variable
    
    z_inf                                  : derived variable
    
    z_tau (ms)                             : derived variable
    
    conductanceScale                       : derived variable
    
    fopen0                                 : derived variable
    
    fopen                                  : derived variable
    
    g (uS)                                 : derived variable
    rate_z_q (/ms)
    
}

STATE {
    z_q  
    
}

INITIAL {
    ek = -75.0
    
    temperature = celsius + 273.15
    
    rates()
    rates() ? To ensure correct initialisation.
    
    z_q = z_inf
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    ? DerivedVariable is based on path: conductanceScaling[*]/factor, on: Component(id=Kahp_pyr type=ionChannelHH), from conductanceScaling; null
    ? Path not present in component, using factor: 1
    
    conductanceScale = 1 
    
    ? DerivedVariable is based on path: gates[*]/fcond, on: Component(id=Kahp_pyr type=ionChannelHH), from gates; Component(id=z type=gateHHrates)
    ? multiply applied to all instances of fcond in: <gates> ([Component(id=z type=gateHHrates)]))
    fopen0 = z_fcond ? path based, prefix = 
    
    fopen = conductanceScale  *  fopen0 ? evaluable
    g = conductance  *  fopen ? evaluable
    gion = gmax * fopen 
    
    ik = gion * (v - ek)
    i__Kahp_pyr =  -1 * ik : set this variable to the current also - note -1 as channel current convention for LEMS used!
    
}

DERIVATIVE states {
    rates()
    z_q' = rate_z_q 
    
}

PROCEDURE rates() {
    LOCAL caConc
    
    caConc = cai
    
    z_forwardRate_V = v /  z_forwardRate_VOLT_SCALE ? evaluable
    z_forwardRate_ca_conc = caConc /  z_forwardRate_CONC_SCALE ? evaluable
    if (z_forwardRate_ca_conc   < ( 500.0 ))  { 
        z_forwardRate_r = ( 0.4 *  z_forwardRate_ca_conc  ) /  z_forwardRate_TIME_SCALE ? evaluable cdv
    } else  { 
        z_forwardRate_r = 200.0  /  z_forwardRate_TIME_SCALE ? evaluable cdv
    }
    
    z_reverseRate_V = v /  z_reverseRate_VOLT_SCALE ? evaluable
    z_reverseRate_ca_conc = caConc /  z_reverseRate_CONC_SCALE ? evaluable
    z_reverseRate_r = 20.0  /  z_reverseRate_TIME_SCALE ? evaluable
    ? DerivedVariable is based on path: q10Settings[*]/q10, on: Component(id=z type=gateHHrates), from q10Settings; null
    ? Path not present in component, using factor: 1
    
    z_rateScale = 1 
    
    ? DerivedVariable is based on path: forwardRate/r, on: Component(id=z type=gateHHrates), from forwardRate; Component(id=null type=Kahp_pyr_z_alpha_rate)
    z_alpha = z_forwardRate_r ? path based, prefix = z_
    
    ? DerivedVariable is based on path: reverseRate/r, on: Component(id=z type=gateHHrates), from reverseRate; Component(id=null type=Kahp_pyr_z_beta_rate)
    z_beta = z_reverseRate_r ? path based, prefix = z_
    
    z_fcond = z_q ^ z_instances ? evaluable
    z_inf = z_alpha /( z_alpha + z_beta ) ? evaluable
    z_tau = 1/(( z_alpha + z_beta ) *  z_rateScale ) ? evaluable
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    rate_z_q = ( z_inf  -  z_q ) /  z_tau ? Note units of all quantities used here need to be consistent!
    
     
    
     
    
     
    
}

