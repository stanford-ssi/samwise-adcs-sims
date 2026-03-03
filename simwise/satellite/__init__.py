from simwise.satellite.data_structures.state import SatelliteState
from simwise.satellite.data_structures.params import SatelliteParams


def __getattr__(name):
    if name == "Satellite":
        from simwise.satellite.satellite import Satellite
        return Satellite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")