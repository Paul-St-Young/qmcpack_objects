import os
import h5py

class PwscfH5:
    def __init__(self):
        self.locations = {
            'gvectors':'electrons/kpoint_0/gvectors'
        }

    def read(self,fname):
        if not os.path.isfile(fname):
            raise RuntimeError('%s not found' % fname)
        self.fp = h5py.File(fname)

    def get(self,name):
        loc = self.locations[name]
        return self.fp[loc].value

    @staticmethod
    def state_path(ikpt,ispin,istate):
        path = 'electrons/kpoint_%d/spin_%d/state_%d/' % (ikpt,ispin,istate)
        return path

    def psig(self,ikpt=0,ispin=0,istate=0):
        psig_loc = self.state_path(ikpt,ispin,istate)+'psi_g'
        return self.fp[psig_loc].value

    def psir(self,ikpt=0,ispin=0,istate=0):
        psir_loc = self.state_path(ikpt,ispin,istate)+'psi_r'
        return self.fp[psir_loc].value

# end class PwscfH5
