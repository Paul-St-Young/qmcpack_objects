import os
import h5py
import numpy as np

class PwscfH5:
    def __init__(self):
        self.locations = {
            'gvectors':'electrons/kpoint_0/gvectors',
            'nkpt':'electrons/number_of_kpoints',
            'nspin':'electrons/number_of_spins',
            'nstate':'electrons/kpoint_0/spin_0/number_of_states' # !!!! same number of states per kpt
        }
        self.dtypes = {
            'gvectors':float,
            'nkpt':int,
            'nspin':int,
            'nstate':int
        }

    def read(self,fname):
        """ open 'fname' for reading and save handle in this class """
        if not os.path.isfile(fname):
            raise RuntimeError('%s not found' % fname)
        self.fp = h5py.File(fname)

    def get(self,name):
        """ get value array of a known entry """
        loc   = self.locations[name]
        dtype = self.dtypes[name]
        return self.fp[loc].value.astype(dtype)

    def val(self,loc):
        """ get value array of an arbitrary entry at location 'loc' """
        return self.fp[loc].value

    @staticmethod
    def spin_path(ikpt,ispin):
        path = 'electrons/kpoint_%d/spin_%d' % (ikpt,ispin)
        return path
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

    def eigenvalues(self):
        """ return all eigenvalues, shape=(nkpt,nspin,nstate) """
        nkpt   = self.get('nkpt')[0]
        nspin  = self.get('nspin')[0]
        nstate = self.get('nstate')[0] # !!!! same number of states per kpt

        evals  = np.zeros([nkpt,nspin,nstate])
        for ikpt in range(nkpt):
            for ispin in range(nspin):
                path = self.spin_path(ikpt,ispin)
                evals[ikpt,ispin,:] = self.val(
                    os.path.join(path,'eigenvalues')
                )
        return evals

# end class PwscfH5
