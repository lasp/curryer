KPL/FK

Frame definitions for the ISS spacecraft & TSIS mission
--------------------------------------------------------

    Frame definitions required for the TSIS mission.

    Frame Name          Relative to Frame   Frame Type  Frame ID
    ==========          =================   ==========  ========

    ISS Frames
    ----------
    ISS_ISSACS          J2000 (ECI)         CK          -125544000
    ISS_ELC3_COORD      ISS_ISSACS          FIXED       -125544030
    ISS_EXPA35_COORD    ISS_ISSACS          FIXED       -125544035

    TSIS Frames
    -----------
    TSIS_TADS_COORD     ISS_EXPA35_COORD    FIXED       -125544109
    TSIS_AZEL_COORD     TSIS_TADS_COORD     CK          -125544108
    TSIS_TIM_COORD      TSIS_AZEL_COORD     FIXED       -125544100


                    TIM      AzEl      TADS    ExPA 3-5    ELC 3   ISSACS     Vernal Equinox        ECI
                    ---      ----      ----    --------    -----   ------     --------------        ---
    *  *            +Y        +Y        +Y        +Z        +Y                     * N *            +Z
 *        *          |         |         |         |         |                  *         *          |
*   Sun    *   +Z -- *   +Z -- *   +Z -- *   +X -- *   +Z -- *       * -- +Z   *   Earth   *   +X -- *
 *        *         /         /         /         /         /       /|          *         *         /
    *  *          +X        +X        +X        +Y        +X     +X +Y             * S *          +Y


    Notes
    -----
    - SPICE matrices are written in column-major order, and must be
    oriented as a rotation *from* Frame *to* Relative.

    Examples
    --------
    >>> spicierpy.furnsh('tsis_v01.frame.fk.tk')
    >>> iss2expa = spicierpy.pxform('ISS_ISSACS', 'ISS_EXPA35_COORD', 0)
    >>> print(iss2expa.dot(np.array([1.0, 0.0, 0.0])).round(2))
    array([ 0.,  1.,  0.])

    References
    ----------

    1. Space Station Reference Coordinate Systems (SSP 30219, Rev J)
    2. ENG MEMO, TSIS, TIM GLINT FOV MECH POSITION (149953, Rev A)
    3. ISS Hardware Coordinate Data, AFRAMS, ELC3 (EID684-12401-123)


    This file was created by LASP_SDS_TEAM
    on 2017-07-20/10:32:00.

Frame offsets
--------------------------------------------------------
    Frame offsets are actually defined in a "static" kernel. The values are
    included here as a reference. Units = meters.

    From Frame          To Frame            Offset [X, Y, Z]
    ==========          ========            ================
    ISS_ISSACS          ISS_EXPA35_COORD    [-2.713,   -25.018,    -6.192]
    ISS_EXPA35_COORD    TSIS_TADS_COORD     [ 0.409575,  0.506349,  0.174625]
    TSIS_TADS_COORD     TSIS_AZEL_COORD     [-0.000004,  0.543928,  0.4318]
    TSIS_AZEL_COORD     TSIS_TIM_COORD      [-0.239377,  0.146924,  0.153198]

Frame definitions
--------------------------------------------------------

    ISS (-125544) - Spacecraft (CK)
    -------------------------------

        \begindata

        FRAME_ISS_ISSACS            = -125544000
        FRAME_-125544000_NAME       = 'ISS_ISSACS'
        FRAME_-125544000_CLASS      = 3
        FRAME_-125544000_CLASS_ID   = -125544000
        FRAME_-125544000_CENTER     = -125544
        CK_-125544000_SCLK          = -125544
        CK_-125544000_SPK           = -125544

        OBJECT_-125544_FRAME        = 'ISS_ISSACS'

        \begintext

    ISS ELC 3 (-125544030) - Structure (TK)
    -----------------------------------------
    Based on the ELC positive Y-offset frame [1].
    TODO: Remove if not implemented. We don't have its offset...

        \begindata

        FRAME_ISS_ELC3_COORD        = -125544030
        FRAME_-125544030_NAME       = 'ISS_ELC3_COORD'
        FRAME_-125544030_CLASS      = 4
        FRAME_-125544030_CLASS_ID   = -125544030
        FRAME_-125544030_CENTER     = -125544
        TKFRAME_-125544030_RELATIVE = 'ISS_ISSACS'
        TKFRAME_-125544030_SPEC     = 'ANGLES'
        TKFRAME_-125544030_ANGLES   = ( 0, 0, 180 )
        TKFRAME_-125544030_AXES     = ( 3, 2,   1 )
        TKFRAME_-125544030_UNITS    = 'DEGREES'

        OBJECT_-125544030_FRAME     = 'ISS_ELC3_COORD'

        \begintext

    ISS ExPA 3-5 (-125544035) - Structure (TK)
    ------------------------------------------
    From ISSACS, this is a ~90 deg rotation about the Y-axis, then ~90 deg
    rotation about the X-axis (i.e., ExPA X-axis ~= ISSACS -Z-axis).

    Note: Rotation includes the rotation to and from the ELC frame.
            ISSACS -> ELC 3 -> ExPA 3-5

        \begindata

        FRAME_ISS_EXPA35_COORD      = -125544035
        FRAME_-125544035_NAME       = 'ISS_EXPA35_COORD'
        FRAME_-125544035_CLASS      = 4
        FRAME_-125544035_CLASS_ID   = -125544035
        FRAME_-125544035_CENTER     = -125544
        TKFRAME_-125544035_RELATIVE = 'ISS_ISSACS'
        TKFRAME_-125544035_SPEC     = 'MATRIX'
        TKFRAME_-125544035_MATRIX   = ( -0.000335889
                                        -0.00147085
                                        -0.999998862
                                         0.999999859
                                        -0.000412046
                                        -0.000335283
                                        -0.000411553
                                        -0.999998833
                                         0.001470988 )

        OBJECT_-125544035_FRAME     = 'ISS_EXPA35_COORD'

        \begintext

        Alternatively defined as angles...
        Assuming ISSACS [roll=X, yaw=Y, pitch=Z], and the rotation order
        [yaw, pitch roll] [3].
        TKFRAME_-125544035_SPEC     = 'ANGLES'
        TKFRAME_-125544035_ANGLES   = ( -90.019, 0.084, -90.024 )
        TKFRAME_-125544035_AXES     = (       2,     3,       1 )
        TKFRAME_-125544035_UNITS    = 'DEGREES'

    TSIS TADS DEPLOYED (-125544109) - TSIS Base (TK)
    ------------------------------------------------

        \begindata

        FRAME_TSIS_TADS_COORD       = -125544109
        FRAME_-125544109_NAME       = 'TSIS_TADS_COORD'
        FRAME_-125544109_CLASS      = 4
        FRAME_-125544109_CLASS_ID   = -125544109
        FRAME_-125544109_CENTER     = -125544
        TKFRAME_-125544109_RELATIVE = 'ISS_EXPA35_COORD'
        TKFRAME_-125544109_SPEC     = 'ANGLES'
        TKFRAME_-125544109_ANGLES   = ( -90, 0, -90 )
        TKFRAME_-125544109_AXES     = (  3,  2,   1 )
        TKFRAME_-125544109_UNITS    = 'DEGREES'

        OBJECT_-125544109_FRAME     = 'TSIS_TADS_COORD'

        \begintext

    TSIS TPS (-125544108) - TPS AzEl (CK)
    -------------------------------------

        \begindata

        FRAME_TSIS_AZEL_COORD       = -125544108
        FRAME_-125544108_NAME       = 'TSIS_AZEL_COORD'
        FRAME_-125544108_CLASS      = 3
        FRAME_-125544108_CLASS_ID   = -125544108
        FRAME_-125544108_CENTER     = -125544109
        CK_-125544108_SCLK          = -125544
        CK_-125544108_SPK           = -125544109

        OBJECT_-125544108_FRAME     = 'TSIS_AZEL_COORD'

        \begintext

    TSIS TIM (-125544100) - TSIS TIM (TK)
    -------------------------------------

        \begindata

        FRAME_TSIS_TIM_COORD        = -125544100
        FRAME_-125544100_NAME       = 'TSIS_TIM_COORD'
        FRAME_-125544100_CLASS      = 4
        FRAME_-125544100_CLASS_ID   = -125544100
        FRAME_-125544100_CENTER     = -125544108
        TKFRAME_-125544100_RELATIVE = 'TSIS_AZEL_COORD'
        TKFRAME_-125544100_SPEC     = 'ANGLES'
        TKFRAME_-125544100_ANGLES   = ( 0,  0,  0 )
        TKFRAME_-125544100_AXES     = ( 3,  2,  1 )
        TKFRAME_-125544100_UNITS    = 'DEGREES'

        OBJECT_-125544100_FRAME     = 'TSIS_TIM_COORD'

        \begintext

    TSIS TIM GLINT (-125544101) - TSIS TIM (TK)
    -------------------------------------

        \begindata

        OBJECT_-125544101_FRAME     = 'TSIS_TIM_COORD'

        \begintext
