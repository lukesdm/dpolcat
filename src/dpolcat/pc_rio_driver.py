# Custom PlanetaryComputer RioDriver for use in odc.stac.load,
# which re-signs assets just before use (rather than up front).
# planetary_computer.sign has a token cache, so it's not overly expensive.
# Code via Claude Opus 5.5.

import dataclasses
from urllib.parse import urlsplit, urlunsplit

import planetary_computer
from odc.loader._rio import RioDriver, RioReader
from odc.stac._mdtools import StacAuxReader


def resign(url: str) -> str:
    return planetary_computer.sign(urlunsplit(urlsplit(url)._replace(query="")))


class PCRioReader(RioReader):
    def read(self, cfg, dst_geobox, *, dst=None, selection=None):
        src = dataclasses.replace(self._src, uri=resign(self._src.uri))
        return RioReader(src, self._ctx).read(
            cfg, dst_geobox, dst=dst, selection=selection
        )


class PCRioDriver(RioDriver):
    def __init__(self):
        super().__init__(aux_reader=StacAuxReader())

    def open(self, src, ctx):
        return PCRioReader(src, ctx)
