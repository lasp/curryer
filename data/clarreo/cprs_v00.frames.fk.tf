KPL/FK

Frame definitions for the ISS spacecraft & CPRS mission
--------------------------------------------------------

    Frame definitions required for the CPRS mission.

    Frame Name          Relative to Frame   Frame Type  Frame ID
    ==========          =================   ==========  ========
    ISS_ISSACS          J2000 (ECI)         CK          -125544000
    CPRS_ST_COORD       J2000 (ECI)         CK          -125544207
    CPRS_BASE_COORD     ISS_ISSACS          FIXED       -125544206
    CPRS_AZEL_COORD     CPRS_BASE_COORD     CK          -125544205
    CPRS_HYSICS_COORD   CPRS_AZEL_COORD     FIXED       -125544200

    TODO: Update figures. Remove "ST" ("star tracker") if unused.

                  HySICS     AzEl      BASE    ExPA ?-?    ELC ?   ISSACS     Vernal Equinox        ECI
                  ------     ----      ----    --------    -----   ------     --------------        ---
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
    >>> spicierpy.furnsh('cprs_v01.frame.fk.tk')
    >>> iss2expa = spicierpy.pxform('ISS_ISSACS', 'ISS_EXPA35_COORD', 0)
    >>> print(iss2expa.dot(np.array([1.0, 0.0, 0.0])).round(2))
    array([ 0.,  1.,  0.])

    References
    ----------

    1. Space Station Reference Coordinate Systems (SSP 30219, Rev J)
    2. Geolocation Sources v11 (citing Agile #159559)


    This file was created by LASP_SDS_TEAM
    on 2023-12-01/00:00:00.

Frame offsets
--------------------------------------------------------
    Frame offsets are actually defined in a "static" kernel. The values are
    included here as a reference. Units = meters.

    From Frame          To Frame            Offset [X, Y, Z]
    ==========          ========            ================
    ISS_ISSACS          CPRS_BASE_COORD     [ 0.27432, -25.397,    6.7970]
    CPRS_ST_COORD       CPRS_BASE_COORD     [ 0.0,       0.0,      0.0]
    CPRS_BASE_COORD     CPRS_AZEL_COORD     [ 0.0,       0.0,      0.0]
    CPRS_AZEL_COORD     CPRS_HYSICS_COORD   [ 0.0,       0.0,      0.0]

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

    CPRS ST (-125544207) - Star Tracker (CK)
    -------------------------------

        \begintext data TODO

        FRAME_CPRS_ST_COORD         = -125544207
        FRAME_-125544207_NAME       = 'CPRS_ST_COORD'
        FRAME_-125544207_CLASS      = 3
        FRAME_-125544207_CLASS_ID   = -125544207
        FRAME_-125544207_CENTER     = -125544
        CK_-125544207_SCLK          = -125544
        CK_-125544207_SPK           = -125544

        OBJECT_-125544207_FRAME        = 'CPRS_ST_COORD'

        \begintext

    CPRS BASE (-125544206) - Structure (TK)
    ------------------------------------------------

        \begindata

        FRAME_CPRS_BASE_COORD       = -125544206
        FRAME_-125544206_NAME       = 'CPRS_BASE_COORD'
        FRAME_-125544206_CLASS      = 4
        FRAME_-125544206_CLASS_ID   = -125544206
        FRAME_-125544206_CENTER     = -125544
        TKFRAME_-125544206_RELATIVE = 'ISS_ISSACS'
        TKFRAME_-125544206_SPEC     = 'ANGLES'
        TKFRAME_-125544206_ANGLES   = (  0,  0,  0 )
        TKFRAME_-125544206_AXES     = (  3,  2,  1 )
        TKFRAME_-125544206_UNITS    = 'DEGREES'

        OBJECT_-125544206_FRAME     = 'CPRS_BASE_COORD'

        \begintext

    CPRS HPS (-125544205) - HPS AzEl (CK)
    -------------------------------------

        \begindata

        FRAME_CPRS_AZEL_COORD       = -125544205
        FRAME_-125544205_NAME       = 'CPRS_AZEL_COORD'
        FRAME_-125544205_CLASS      = 3
        FRAME_-125544205_CLASS_ID   = -125544205
        FRAME_-125544205_CENTER     = -125544206
        CK_-125544205_SCLK          = -125544
        CK_-125544205_SPK           = -125544206

        OBJECT_-125544205_FRAME     = 'CPRS_AZEL_COORD'

        \begintext

    CPRS HySICS (-125544200) - CPRS HySICS (TK)
    -------------------------------------

        \begindata

        FRAME_CPRS_HYSICS_COORD     = -125544200
        FRAME_-125544200_NAME       = 'CPRS_HYSICS_COORD'
        FRAME_-125544200_CLASS      = 4
        FRAME_-125544200_CLASS_ID   = -125544200
        FRAME_-125544200_CENTER     = -125544205
        TKFRAME_-125544200_RELATIVE = 'CPRS_AZEL_COORD'
        TKFRAME_-125544200_SPEC     = 'ANGLES'
        TKFRAME_-125544200_ANGLES   = ( 0,  0,  0 )
        TKFRAME_-125544200_AXES     = ( 3,  2,  1 )
        TKFRAME_-125544200_UNITS    = 'DEGREES'

        OBJECT_-125544200_FRAME     = 'CPRS_HYSICS_COORD'

        \begintext
