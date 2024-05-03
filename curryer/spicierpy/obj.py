# pylint: disable=unbalanced-tuple-unpacking
"""SPICE (NAIF) ID/Name objects.

Examples
--------
After loading the mission kernels:
>>> sc = Spacecraft('tcte', infer_all=True)
>>> print(sc.to_string(verbose=True))
Spacecraft('TCTE' or -1234)
    Frame('TCTE_SC_COORD' or -1234000)
    Clock('TCTE' or -1234)
    Ephemeris('TCTE' or -1234)
    Attitude('-1234000' or -1234000)
    Instrument('TCTE_TIM' or -1234001)
        Frame('TCTE_TIM' or -1234001)
    Instrument('TCTE_TIM_GLINT' or -1234011)
        Frame('TCTE_TIM' or -1234001)

>>> print(sc.frame, sc.get_instrument('tcte_tim').frame)
Frame(TCTE_SC_COORD) Frame(TCTE_TIM)


Notes
-----
- Supports:
    - General NAIF object ID system.
    - Frame ID system.
- Does not support:
    - Surface ID system.

@author: Brandon Stone
"""
import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import spiceypy
import spiceypy.utils.exceptions


logger = logging.getLogger(__name__)

_INDENT = '    '


class AbstractObj(metaclass=ABCMeta):
    """Light-weight abstract class for storing SPICE name <--> id mappings.
    """
    __slots__ = ['_name', '_id']

    def __init__(self, id_or_name):
        """Create a mapping object using an ID or name.

        Parameters
        ----------
        id_or_name : int or str or AbstractObj
            NAIF object ID (int) or name (str). The if the name is not already
            registered, use `define(name, id_code)`, otherwise the name must be
            convertible to an int (e.g., int("-1234") == -1234).

        Notes
        -----
        - Names are converted to uppercase with leading and trailing spaces
        removed.

        """
        if isinstance(id_or_name, AbstractObj):
            self._id = id_or_name.id
            self._name = id_or_name.name
        elif isinstance(id_or_name, str):
            self._id = self._name2code(id_or_name)
            self._name = id_or_name.strip().upper()
        else:
            id_or_name = int(id_or_name)
            self._name = self._code2name(id_or_name).strip().upper()
            self._id = id_or_name

    @classmethod
    @abstractmethod
    def define(cls, name, id_code, **kwargs):
        """Register a new name <--> ID mapping and return its object.

        Parameters
        ----------
        name : str
        id_code : int
        **kwargs
            Extra keywords are passed to init.

        Returns
        -------
        AbstractObj

        Notes
        -----
        - This mapping has a medium priority; above built-in mappings, but
        below text kernel mappings.
        - Multiple names can be mapped to the same ID; last-in, first-out.
        - Frame mappings must be defined in a frame kernel (FK).

        """
        raise NotImplementedError

    def __repr__(self):
        """<class-name>(<obj-name> or <obj-id>)
        """
        return self.to_string().rstrip()

    def __eq__(self, other):
        """Equate objects if one is a child of the other and their IDs match.
        """
        return (isinstance(other, self.__class__) or isinstance(self, type(other))) and self.id == other.id

    @property
    def name(self):
        """Object name (uppercase).

        Returns
        -------
        str

        """
        return self._name

    @property
    def id(self):
        """Object ID.

        Returns
        -------
        int

        """
        return self._id

    @abstractmethod
    def _name2code(self, value):
        """Return the SPICE ID for to the given SPICE name.
        """
        raise NotImplementedError

    @abstractmethod
    def _code2name(self, value):
        """Return the SPICE name for to the given SPICE ID.
        """
        raise NotImplementedError

    def to_string(self, depth=0, verbose=False):
        """String representation with an optional indent level.
        """
        if verbose:
            fmt = '{indent}{self.__class__.__name__}({self.name!r} or {self.id})\n'
        else:
            fmt = '{self.__class__.__name__}({self.name})'
        return fmt.format(indent=_INDENT * depth, self=self)

    def _check_if_missing(self, value, attr_name):
        """Raise an error if missing a value.
        """
        if value is None:
            raise ValueError('{} does not have an attached {!r}'.format(self, attr_name))
        return value


class Body(AbstractObj):
    """General class for SPICE object mapping.

    Examples
    --------
    >>> earth = Body('earth')
    >>> print(earth.name, earth.id)
    EARTH 399

    >>> print(earth, earth.frame)
    Body('EARTH' or 399) Frame('ITRF93' or 13000)

    """
    __slots__ = ['_frame']

    def __init__(self, id_or_name, frame=None):
        super().__init__(id_or_name)
        self._frame = None
        self.frame = frame

    @classmethod
    def define(cls, name, id_code, **kwargs):
        spiceypy.boddef(name, id_code)
        return cls(name, **kwargs)

    @property
    def frame(self):
        """SPICE frame related to this body (e.g., SUN --> IAU_SUN).

        Returns
        -------
        Frame

        """
        return self._check_if_missing(self._frame, 'Frame')

    @frame.setter
    def frame(self, value):
        if value is None or isinstance(value, Frame):
            pass
        elif value is True:
            logger.debug('Looking up FRAME ID from BODY=[%s]', self.id)
            frame_id, _ = spiceypy.cidfrm(self.id)
            value = Frame(frame_id, body=self)
        else:
            value = Frame(value, body=self)
        self._frame = value

    _name2code = staticmethod(spiceypy.bods2c)  # bodn2c  <-- Errors if no name. Hmmm.
    _code2name = staticmethod(spiceypy.bodc2s)  # bodc2n


class Frame(AbstractObj):
    """Frame mapping for default and FK-based mappings.

    Examples
    --------
    >>> iau_sun = Frame('iau_sun')
    >>> print(iau_sun.name, iau_sun.id)
    IAU_SUN 10010

    >>> print(iau_sun, iau_sun.body)
    Frame('IAU_SUN' or 10010) Body('SUN' or 10)

    Notes
    -----
    - The frame ID system is separate from the general SPICE object system, and
    the same ID value is interpreted differently depending on the system.
        - Generally instrument or spacecraft subsystem ID vs. frame ID.

    """
    __slots__ = ['_body']

    def __init__(self, id_or_name, body=None):
        super().__init__(id_or_name)
        self._body = None
        self.body = body

    @classmethod
    def define(cls, name, id_code, **kwargs):
        frame = cls(id_code, **kwargs)
        if frame.name != name:
            # Limitation of SPICE. TODO: Improve?
            # pylint: disable=protected-access
            logger.warning('Frame name <--> id mapping must be defined in a frame kernel.'
                           ' Assuming the kernel is loaded at a later point.')
            frame._name = name
            # raise NotImplementedError('Frame name <--> id mapping must be defined in a frame kernel.')
        return frame

    @property
    def body(self):
        """SPICE body related to this frame (e.g., IAU_SUN --> SUN).

        Returns
        -------
        Body

        """
        return self._check_if_missing(self._body, 'Body')

    @body.setter
    def body(self, value):
        if value is None or isinstance(value, Body):
            pass
        elif value is True:
            logger.debug('Looking up BODY ID from FRAME=[%s]', self.id)
            # Note: `frinfo` --> frame center, frame class (type), frame class id.
            frame_center_id, _, _ = spiceypy.frinfo(self.id)
            value = Body(frame_center_id, frame=self)
        else:
            value = Body(value, frame=self)
        self._body = value

    def _name2code(self, value):
        code = spiceypy.namfrm(value)
        return code if code != 0 else str(value)

    def _code2name(self, value):
        name = spiceypy.frmnam(value)
        return name if name != '' else str(value)


class AnyBodyOrFrame(AbstractObj):
    """Obj representing a body or frame.
    """
    __slots__ = ['_is_frame', '_obj']

    def __init__(self, id_or_name, body=True, frame=True):
        try:
            self._obj = Body(id_or_name, frame=frame)
            self._is_frame = False
        except spiceypy.utils.exceptions.NotFoundError:
            self._obj = Frame(id_or_name, body=body)
            self._is_frame = True
        super().__init__(self._obj)

    @property
    def is_frame(self):
        return self._is_frame

    @property
    def body(self):
        return self._obj.body if self.is_frame else self._obj

    @property
    def frame(self):
        return self._obj if self.is_frame else self._obj.frame

    @classmethod
    def define(cls, name, id_code, **kwargs):
        raise NotImplementedError

    def _name2code(self, value):
        assert value == self._name
        return self._id

    def _code2name(self, value):
        assert value == self._id
        return self._name


class Spacecraft(Body):
    """Special case where the object has a clock, ephemeris, attitude, frame
    and zero or more instruments; typically a spacecraft.

    Examples
    --------
    >>> sc = Spacecraft('tcte')
    >>> print(sc.name, sc.id)
    TCTE -1234

    >>> print(sc.id, sc.attitude.id, sc.frame.id)
    -1234 -1234000 -1234000

    >>> print(sc.get_instrument('tcte_tim'))
    Instrument('TCTE_TIM' or -1234001)
        Frame('TCTE_TIM' or -1234001)

    """
    __slots__ = ['_clock', '_ephemeris', '_attitude', '_instruments']

    def __init__(self, id_or_name, frame=None, clock=None, ephemeris=None, attitude=None, instruments=None,
                 infer_all=False):
        if infer_all:
            frame = clock = ephemeris = attitude = instruments = True
        super().__init__(id_or_name, frame=frame)
        self._clock = None
        self._ephemeris = None
        self._attitude = None
        self._instruments = None
        self.clock = clock
        self.ephemeris = ephemeris
        self.attitude = attitude
        self.instruments = instruments

    def get_instrument(self, id_or_name):
        """Retrieve an instrument mapping and associate it with this
        spacecraft mapping.

        Parameters
        ----------
        id_or_name : int or str
            Instrument ID or name.

        Returns
        -------
        Instrument

        """
        instrument = Instrument(id_or_name, spacecraft=self)
        if instrument.id not in self.instruments:
            self._instruments[instrument.id] = instrument
        return self._instruments[instrument.id]

    @property
    def instruments(self):
        """All SPICE objects with an ID in the standard instrument ID range.

        Returns
        -------
        collections.OrderedDict
            Instrument IDs and Instrument mappings related to this spacecraft.

        Notes
        -----
        - Standard instrument ID range:
            [<spacecraft-id> * 1000, <spacecraft-id> * 1000 - 999]
        - Based on the kernel pool variable: "NAIF_BODY_CODE"

        """
        return self._instruments

    @instruments.setter
    def instruments(self, value):
        if isinstance(value, dict):
            pass
        elif value is None:
            value = OrderedDict()
        elif not (value is True or isinstance(value, list)):
            raise TypeError('`instruments` must be None, True, dict or list, not: {}'.format(value))

        else:
            # Infer instruments...
            if value is True:
                codes = spiceypy.gipool('NAIF_BODY_CODE', 0, 2000)
                base_id = self.id * 1000
                instrument_codes = codes[(codes // base_id == 1) & (codes % base_id > -1000)]
            else:
                instrument_codes = value.copy()

            value = OrderedDict()
            for instr in instrument_codes:
                if not isinstance(instr, Instrument):
                    instr = Instrument(instr, frame=True, spacecraft=self)
                value[instr.id] = instr

        self._instruments = value

    @property
    def clock(self):
        """Spacecraft mapping object for this spacecraft.

        Returns
        -------
        Clock
            Assumes the clock uses the same ID as the spacecraft (typical).

        """
        return self._check_if_missing(self._clock, 'Clock')

    @clock.setter
    def clock(self, value):
        if value is None or isinstance(value, Clock):
            pass
        elif value is True:
            # kvar_name = 'CK_{}_SCLK'.format(frame_class_id)
            # if spiceypy.expool(kvar_name):
            #     sclk_id = spiceypy.gipool(kvar_name, 0, 1).tolist()[0]
            value = Clock(self.id, spacecraft=self)
        else:
            value = Clock(value, spacecraft=self)
        self._clock = value

    @property
    def ephemeris(self):
        """Ephemeris mapping for this spacecraft.

        Returns
        -------
        Ephemeris
            Assumes the SPK uses the same ID as the spacecraft (typical).

        """
        return self._check_if_missing(self._ephemeris, 'Ephemeris')

    @ephemeris.setter
    def ephemeris(self, value):
        if value is None or isinstance(value, Ephemeris):
            pass
        elif value is True:
            value = Ephemeris(self.id, spacecraft=self)
        else:
            value = Ephemeris(value, spacecraft=self)
        self._ephemeris = value

    @property
    def attitude(self):
        """Attitude mapping for this spacecraft.

        Returns
        -------
        Attitude
            Attitude mapped to the spacecrafts reference frame; must use a
            CK-based reference frame spacecraft.

        """
        return self._check_if_missing(self._attitude, 'Attitude')

    @attitude.setter
    def attitude(self, value):
        if value is None or isinstance(value, Attitude):
            pass
        elif value is True:
            _, frame_class, frame_class_id = spiceypy.frinfo(self.frame.id)
            if frame_class != 3:
                pass
            value = Attitude(frame_class_id, spacecraft=self)
        else:
            value = Attitude(value, spacecraft=self)
        self._attitude = value

    def to_string(self, depth=0, verbose=False):
        """String representation with an optional indent level.
        """
        self_txt = super().to_string(depth=depth, verbose=verbose)
        if verbose:
            self_txt = '{self}{frame}{clock}{ephemeris}{attitude}{instruments}'.format(
                self=self_txt,
                frame=self.frame.to_string(depth=depth + 1, verbose=True),
                clock=self.clock.to_string(depth=depth + 1, verbose=True),
                ephemeris=self.ephemeris.to_string(depth=depth + 1, verbose=True),
                attitude=self.attitude.to_string(depth=depth + 1, verbose=True),
                instruments=''.join(
                    instr.to_string(depth=depth + 1, verbose=True) for instr in self.instruments.values()
                )
            )
        return self_txt


class _SpacecraftItem(Body):
    """Abstract mapping class that can store a reference to a spacecraft
    mapping object.
    """
    __slots__ = ['_spacecraft']

    def __init__(self, id_or_name, frame=None, spacecraft=None):
        """Create a mapping object using an ID or name. May also include a
        spacecraft object.

        Parameters
        ----------
        id_or_name : int or str or _SpacecraftItem
            NAIF object ID (int) or name (str). The if the name is not already
            registered, use `define(name, id_code)`, otherwise the name must be
            convertible to an int (e.g., int("-1234001") == -1234001).
        spacecraft : Spacecraft or True, optional
            Optional Spacecraft that this item is related to. Default is None,
            and makes an assumption about the spacecraft's ID. If `id_or_name`
            is a `_SpacecraftItem` and the `spacecraft` keyword is None, then
            its value is taken from `id_or_name.spacecraft`.

        """
        super().__init__(id_or_name, frame=frame)
        self._spacecraft = None
        self.spacecraft = spacecraft

    def _infer_spacecraft(self, id_code):
        """Infer the spacecraft from the object's ID.
        """
        raise NotImplementedError

    @property
    def spacecraft(self):
        """Spacecraft mapping related to this item.

        Returns
        -------
        Spacecraft
            Either the spacecraft supplied during init (e.g., item accessed
            through a Spacecraft instance), or a new instance based ID
            assumptions (see subclass method `_infer_spacecraft_id`).

        """
        return self._check_if_missing(self._spacecraft, 'Spacecraft')

    @spacecraft.setter
    def spacecraft(self, value):
        if value is None or isinstance(value, Spacecraft):
            pass
        elif value is True:
            value = self._infer_spacecraft(self.id)
        else:
            value = Spacecraft(value)
        self._spacecraft = value


class Instrument(_SpacecraftItem):
    """Instrument mapping.

    Examples
    --------
    >>> tim = Instrument('tcte_tim')
    >>> print(tim.name, tim.id)
    TCTE_TIM -1234001

    >>> print(tim.spacecraft.name, tim.frame)
    TCTE Frame('TCTE_TIM' or -1234001)

    """
    __slots__ = []

    def _infer_spacecraft(self, id_code):
        id_code = int(np.ceil(id_code / 1000))  # E.g., -1234001 --> -1234
        return Spacecraft(id_code, instruments=[self])

    def to_string(self, depth=0, verbose=False):
        """String representation with an optional indent level.
        """
        self_txt = super().to_string(depth=depth, verbose=verbose)
        if verbose:
            self_txt = '{self}{frame}'.format(
                self=self_txt,
                frame=self.frame.to_string(depth=depth + 1, verbose=True)
            )
        return self_txt


class Clock(_SpacecraftItem):
    """Spacecraft clock mapping.

    Examples
    --------
    >>> sclk = Clock('tcte')
    >>> print(sclk.name, sclk.id)
    TCTE -1234

    """
    __slots__ = []

    def _infer_spacecraft(self, id_code):
        return Spacecraft(id_code, clock=self)  # E.g., They are the same.


class Ephemeris(_SpacecraftItem):
    """Spacecraft ephemeris (SPK) mapping.

    Examples
    --------
    >>> spk = Ephemeris('tcte')
    >>> print(spk.name, spk.id)
    TCTE -1234

    """
    __slots__ = []

    def _infer_spacecraft(self, id_code):
        return Spacecraft(id_code, ephemeris=self)  # E.g., They are the same.


class Attitude(_SpacecraftItem):
    """Spacecraft attitude (CK) mapping.

    Examples
    --------
    >>> ck = Attitude(-1234000)
    >>> print(repr(ck.name), ck.id)
    '-1234000' -1234000

    >>> print(ck.spacecraft.name, ck.spacecraft.id)
    TCTE -1234

    """
    __slots__ = []

    def _infer_spacecraft(self, id_code):
        id_code = int(np.ceil(id_code / 1000))  # E.g., -1234000 --> -1234
        return Spacecraft(id_code, attitude=self)
