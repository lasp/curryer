\begintext

Recall that the frame `EARTH_FIXED' is a TK frame.  As a result
its relationship to other frames must be specified via
a kernel pool variable.  We make that specification here.

    We use IAU_EARTH for now because that is always available.

    \begintext

    TKFRAME_EARTH_FIXED_RELATIVE = 'IAU_EARTH'
    TKFRAME_EARTH_FIXED_SPEC     = 'MATRIX'
    TKFRAME_EARTH_FIXED_MATRIX   = ( 1   0   0
                                     0   1   0
                                     0   0   1 )



    Use the definition below if you have a high precision ITRF93 binary
    pck file.

    \begindata

    TKFRAME_EARTH_FIXED_RELATIVE = 'ITRF93'
    TKFRAME_EARTH_FIXED_SPEC     = 'MATRIX'
    TKFRAME_EARTH_FIXED_MATRIX   = ( 1   0   0
                                     0   1   0
                                     0   0   1 )

