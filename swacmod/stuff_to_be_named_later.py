from dataclasses import dataclass
import multiprocessing as mp

# TODO We need a better name for this stuff.

@dataclass
class Some_Stuff:
    reporting_agg: object
    reporting_agg2: object
    reporting: object
    spatial: object
    single_node_output: object

def make_single_threaded_stuff():
    return Some_Stuff({}, {}, {}, {}, {})

def make_multiprocessing_stuff():
    manager = mp.Manager()
    return Some_Stuff(
        manager.dict(),
        manager.dict(),
        manager.dict(),
        manager.dict(),
        manager.dict())

