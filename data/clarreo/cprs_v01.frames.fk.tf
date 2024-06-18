KPL/FK

Frame definitions for the ISS spacecraft & CPRS mission
--------------------------------------------------------

    Frame definitions required for the CPRS mission.

    Frame Name          Relative to Frame   Frame Type  Frame ID
    ==========          =================   ==========  ========
    ISS_ISSACS          J2000 (ECI)         CK          -125544000
    CPRS_ST_COORD       J2000 (ECI)         CK          -125544209
    CPRS_BASE_COORD     ISS_ISSACS          FIXED       -125544208
    CPRS_PEDE_COORD     CPRS_BASE_COORD     FIXED       -125544207
    CPRS_AZ_COORD       CPRS_PEDE_COORD     CK          -125544206
    CPRS_YOKE_COORD     CPRS_AZ_COORD       FIXED       -125544205
    CPRS_EL_COORD       CPRS_YOKE_COORD     CK          -125544204
    CPRS_HYSICS_COORD   CPRS_EL_COORD       FIXED       -125544200

    TODO: Update figures and remove star tracker items if unused.

                  HySICS      El       YOKE       Az       PEDE     BASE   ISSACS     Vernal Equinox        ECI
                  ------      --       ----       --       ----     ----   ------     --------------        ---
    *  *            +Y        +Y        +Y        +Y        +Y       +Y                    * N *            +Z
 *        *          |         |         |         |         |        |                 *         *          |
*   Sun    *   +Z -- *   +Z -- *   +Z -- *   +Z -- *   +Z -- *  +Z -- *      * -- +Z   *   Earth   *   +X -- *
 *        *         /         /         /         /         /        /      /|          *         *         /
    *  *          +X        +X        +X        +X        +X       +X    +X +Y             * S *          +Y


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
    CPRS_BASE_COORD     CPRS_PEDE_COORD     [ 0.0,       0.0,      0.0]
    CPRS_PEDE_COORD     CPRS_AZ_COORD       [ 0.0,       0.0,      0.0]
    CPRS_AZ_COORD       CPRS_YOKE_COORD     [ 0.0,       0.0,      0.0]
    CPRS_YOKE_COORD     CPRS_EL_COORD       [ 0.0,       0.0,      0.0]
    CPRS_EL_COORD       CPRS_HYSICS_COORD   [ 0.0,       0.0,      0.0]

    TODO: Add back to offset kernel once in prod-mode?
    "cprs_base_v01.fixed_offset.spk.bsp",
    "cprs_pede_v01.fixed_offset.spk.bsp",
    "cprs_az_v01.fixed_offset.spk.bsp",
    "cprs_yoke_v01.fixed_offset.spk.bsp",
    "cprs_el_v01.fixed_offset.spk.bsp",
    "cprs_hysics_v01.fixed_offset.spk.bsp"

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

    CPRS ST (-125544209) - Star Tracker (CK)
    -------------------------------

        \begintext data TODO

        FRAME_CPRS_ST_COORD         = -125544209
        FRAME_-125544209_NAME       = 'CPRS_ST_COORD'
        FRAME_-125544209_CLASS      = 3
        FRAME_-125544209_CLASS_ID   = -125544209
        FRAME_-125544209_CENTER     = -125544
        CK_-125544209_SCLK          = -125544
        CK_-125544209_SPK           = -125544

        OBJECT_-125544209_FRAME        = 'CPRS_ST_COORD'

        \begintext

    CPRS BASE (-125544208) - Structure (TK)
    ------------------------------------------------

        \begindata

        FRAME_CPRS_BASE_COORD       = -125544208
        FRAME_-125544208_NAME       = 'CPRS_BASE_COORD'
        FRAME_-125544208_CLASS      = 4
        FRAME_-125544208_CLASS_ID   = -125544208
        FRAME_-125544208_CENTER     = -125544
        TKFRAME_-125544208_RELATIVE = 'ISS_ISSACS'
        TKFRAME_-125544208_SPEC     = 'MATRIX'
        TKFRAME_-125544208_MATRIX   = (  0.999976773
                                         0.004372441
                                         0.005228291
                                        -0.004363249
                                         0.999988918
                                        -0.001768158
                                        -0.005235964
                                         0.001745304
                                         0.999984769 )

        OBJECT_-125544208_FRAME     = 'CPRS_BASE_COORD'

        \begintext

    CPRS PEDE (-125544207) - Structure (TK)
    ------------------------------------------------

        \begindata

        FRAME_CPRS_PEDE_COORD       = -125544207
        FRAME_-125544207_NAME       = 'CPRS_PEDE_COORD'
        FRAME_-125544207_CLASS      = 4
        FRAME_-125544207_CLASS_ID   = -125544207
        FRAME_-125544207_CENTER     = -125544208
        TKFRAME_-125544207_RELATIVE = 'CPRS_BASE_COORD'
        TKFRAME_-125544207_SPEC     = 'MATRIX'
        TKFRAME_-125544207_MATRIX   = (  0.999986231
                                        -0.000349218
                                        -0.000174228
                                         0.000349061
                                         0.999999558
                                        -0.000872725
                                         0.000174533
                                         0.000872665
                                         0.999999604 )

        OBJECT_-125544207_FRAME     = 'CPRS_PEDE_COORD'

        \begintext

    CPRS HPS (-125544206) - HPS AzEl (CK)
    -------------------------------------

        \begindata

        FRAME_CPRS_AZ_COORD         = -125544206
        FRAME_-125544206_NAME       = 'CPRS_AZ_COORD'
        FRAME_-125544206_CLASS      = 3
        FRAME_-125544206_CLASS_ID   = -125544206
        FRAME_-125544206_CENTER     = -125544207
        CK_-125544206_SCLK          = -125544
        CK_-125544206_SPK           = -125544207

        OBJECT_-125544206_FRAME     = 'CPRS_AZ_COORD'

        \begintext

    CPRS YOKE (-125544205) - Structure (TK)
    ------------------------------------------------

        \begindata

        FRAME_CPRS_YOKE_COORD       = -125544205
        FRAME_-125544205_NAME       = 'CPRS_YOKE_COORD'
        FRAME_-125544205_CLASS      = 4
        FRAME_-125544205_CLASS_ID   = -125544205
        FRAME_-125544205_CENTER     = -125544206
        TKFRAME_-125544205_RELATIVE = 'CPRS_AZ_COORD'
        TKFRAME_-125544205_SPEC     = 'MATRIX'
        TKFRAME_-125544205_MATRIX   = (  0.999999970
                                        -0.000174691
                                         0.000174374
                                         0.000174533
                                         0.999999573
                                         0.000907602
                                        -0.000174533
                                        -0.000907571
                                         0.999999573 )

        OBJECT_-125544205_FRAME     = 'CPRS_YOKE_COORD'

        \begintext

    CPRS HPS Elevation (-125544204) - HPS El (CK)
    -------------------------------------

        \begindata

        FRAME_CPRS_EL_COORD         = -125544204
        FRAME_-125544204_NAME       = 'CPRS_EL_COORD'
        FRAME_-125544204_CLASS      = 3
        FRAME_-125544204_CLASS_ID   = -125544204
        FRAME_-125544204_CENTER     = -125544205
        CK_-125544204_SCLK          = -125544
        CK_-125544204_SPK           = -125544205

        OBJECT_-125544204_FRAME     = 'CPRS_EL_COORD'

        \begintext

    CPRS HySICS (-125544200) - CPRS HySICS (TK)
    -------------------------------------

        \begindata

        FRAME_CPRS_HYSICS_COORD     = -125544200
        FRAME_-125544200_NAME       = 'CPRS_HYSICS_COORD'
        FRAME_-125544200_CLASS      = 4
        FRAME_-125544200_CLASS_ID   = -125544200
        FRAME_-125544200_CENTER     = -125544204
        TKFRAME_-125544200_RELATIVE = 'CPRS_EL_COORD'
        TKFRAME_-125544200_SPEC     = 'MATRIX'
        TKFRAME_-125544200_MATRIX   = (  0.999999859
                                        -0.000401669
                                        -0.000348785
                                         0.000401426
                                         0.999999676
                                        -0.000698272
                                         0.000349066
                                         0.000698132
                                         0.999999695 )

        OBJECT_-125544200_FRAME     = 'CPRS_HYSICS_COORD'

        \begintext
