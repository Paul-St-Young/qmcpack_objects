import h5py

class H5Conf:
    def __init__(self):
        self.paths = {
            'walkers':'state_0/walkers',
            'nwalker':'state_0/number_of_walkers'
        }
    # end def

    def read(self,h5file):
        self.fp = h5py.File(h5file)
    # end def

    def get_entry(self,name):
        if name not in self.paths.keys():
            raise RuntimeError('\'%s\' not found in %s'% (name,str(self.paths.keys())) )
        return self.fp[self.paths['walkers']].value
    # end def
# class H5Conf
