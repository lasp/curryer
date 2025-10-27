"""SPICE Meta-kernel replacement logic.

@author: Brandon Stone
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import spicierpy

logger = logging.getLogger(__name__)

SC_FIELDS = {"clock", "attitude", "ephemeris", "instruments", "infer_all"}


@dataclass(repr=False)
class MetaKernel:
    """SPICE meta-kernel properties."""

    mappings: dict[str, spicierpy.obj.AbstractObj] = field(default_factory=dict)
    mission_kernels: list[Path] = field(default_factory=list, repr=False)
    sds_kernels: list[Path] = field(default_factory=list, repr=False)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(str(val) for val in self.mappings.values())})"

    def first(self):
        try:
            return next(iter(self.mappings.values()))
        except StopIteration:
            return None

    def last(self):
        val = None
        for val in self.mappings.values():
            pass
        return val

    def find_frame_kernels(self) -> list[Path]:
        return [fn for fn in self.mission_kernels if fn.suffix == ".tf"]

    def load(self):
        return spicierpy.ext.load_kernel([self.mission_kernels, self.sds_kernels])

    @staticmethod
    def _render_paths(files, rel_dir: Path):
        output = []
        if not files:
            return output

        for fn in files:
            fn = Path(fn)
            if not fn.is_absolute() and rel_dir is not None:
                fn = rel_dir / fn
            output.append(fn)

            if not fn.is_file():
                logger.warning("Missing file [%s] defined in meta-kernel!", fn)
        return output

    @classmethod
    def from_dict(cls, properties, mission_dir: Path = None, sds_dir: Path = None):
        if mission_dir is None:
            mission_dir = Path.cwd()
        if sds_dir is None:
            sds_dir = Path.cwd()

        properties["mission_kernels"] = cls._render_paths(properties.get("mission_kernels", []), rel_dir=mission_dir)
        properties["sds_kernels"] = cls._render_paths(properties.get("sds_kernels", []), rel_dir=sds_dir)

        frame_kernels = [fn for fn in properties["mission_kernels"] if fn.suffix == ".tf"]
        loaded_krns = spicierpy.ext.load_kernel(frame_kernels) if frame_kernels else None

        mappings = {}
        for name, attr in properties["mappings"].items():
            if not isinstance(attr, dict):
                mappings[name] = spicierpy.obj.Body.define(name, attr)
            elif SC_FIELDS.intersection(attr):
                mappings[name] = spicierpy.obj.Spacecraft.define(name=name, **attr)
            else:
                mappings[name] = spicierpy.obj.Body.define(name=name, **attr)
        properties["mappings"] = mappings

        if loaded_krns is not None:
            loaded_krns.unload(clear=True)

        return cls(**properties)

    @classmethod
    def from_json(cls, json_file, relative: bool = False, mission_dir: Path = None, sds_dir: Path = None):
        json_file = Path(json_file)
        txt = json_file.read_text()
        properties = json.loads(txt)

        if relative:
            if mission_dir is None:
                mission_dir = json_file.parent
            if sds_dir is None:
                sds_dir = json_file.parent

        return cls.from_dict(properties, mission_dir=mission_dir, sds_dir=sds_dir)
