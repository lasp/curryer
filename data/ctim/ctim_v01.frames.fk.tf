KPL/FK

Frame definitions for the CTIM mission & spacecraft
--------------------------------------------------------

    Frame definitions required for the CTIM mission.

    Frame Name          Relative to Frame   Frame Type  Frame ID
    ==========          =================   ==========  ========
    CTIM_COORD          J2000 (ECI)         CK          -152950000
    CTIM_TIM_COORD      CTIM_COORD          FIXED       -152950100


                    TIM      CTIM  Vernal Equinox        ECI
                    ---      ----  --------------        ---
    *  *            +Y        +Y        * N *            +Z
 *        *          |         |     *         *          |
*   Sun    *   -Z -- *   -Z -- *    *   Earth   *   +X -- *
 *        *         /         /      *         *         /
    *  *          +X        +X          * S *          +Y


    Notes
    -----
    - SPICE matrices are written in column-major order, and must be
    oriented as a rotation *from* Frame *to* Relative.

    Examples
    --------
    >>> spicierpy.furnsh('ctim_v01.frame.fk.tk')
    >>> iss2expa = spicierpy.pxform('CTIM_COORD', 'CTIM_TIM_COORD', 0)
    >>> print(iss2expa.dot(np.array([1.0, 0.0, 0.0])).round(2))
    array([ 0.,  1.,  0.])

    References
    ----------

    1. ???


    This file was created by LASP_SDS_TEAM
    on 2022-07-20/10:32:00.

Frame offsets
--------------------------------------------------------
    Frame offsets are actually defined in a "static" kernel. The values are
    included here as a reference. Units = meters.

    From Frame          To Frame            Offset [X, Y, Z]
    ==========          ========            ================
    CTIM                CTIM_TIM_COORD      [ 0.0,       0.0,       0.0]

Frame definitions
--------------------------------------------------------

    ISS (-152950) - CTIM Spacecraft (CK)
    -------------------------------

        \begindata

        FRAME_CTIM_COORD            = -152950000
        FRAME_-152950000_NAME       = 'CTIM_COORD'
        FRAME_-152950000_CLASS      = 3
        FRAME_-152950000_CLASS_ID   = -152950000
        FRAME_-152950000_CENTER     = -152950
        CK_-152950000_SCLK          = -152950
        CK_-152950000_SPK           = -152950

        OBJECT_-152950_FRAME        = 'CTIM_COORD'

        \begintext

    CTIM TIM (-152950100) - CTIM TIM (TK)
    -------------------------------------

        \begindata

        FRAME_CTIM_TIM_COORD        = -152950100
        FRAME_-152950100_NAME       = 'CTIM_TIM_COORD'
        FRAME_-152950100_CLASS      = 4
        FRAME_-152950100_CLASS_ID   = -152950100
        FRAME_-152950100_CENTER     = -152950
        TKFRAME_-152950100_RELATIVE = 'CTIM_COORD'
        TKFRAME_-152950100_SPEC     = 'QUATERNION'
        TKFRAME_-152950100_Q        = ( 9.999876e-01,
                                        1.534418e-03,
                                        4.728365e-03,
                                       -7.255375e-06 )

        OBJECT_-152950100_FRAME     = 'CTIM_TIM_COORD'

        \begintext

        First updated set of quaternions (replaced):
            TKFRAME_-152950100_Q        = ( 9.999609e-01,
                                           -8.690019e-03,
                                           -1.641565e-03,
                                            1.426579e-05 )

        Second updated set of quats. TODO: Reverted?
            TKFRAME_-152950100_Q        = ( 9.999872e-01,
                                           -3.708777e-04,
                                           -5.048343e-03,
                                            1.872342e-06 )

        Misc testing...
            TKFRAME_-152950100_Q        = ( 9.999607e-01,
                                           -8.722743e-03,
                                            1.582371e-03,
                                           -1.380316e-05 )
