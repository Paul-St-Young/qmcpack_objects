import os
import h5py
import numpy as np

class PwscfH5:
    def __init__(self):
        self.locations = {
            'gvectors':'electrons/kpoint_0/gvectors',
            'nkpt':'electrons/number_of_kpoints',
            'nspin':'electrons/number_of_spins',
            'nstate':'electrons/kpoint_0/spin_0/number_of_states', # !!!! same number of states per kpt
            'axes':'supercell/primitive_vectors'
        }
        self.dtypes = {
            'nkpt':int,
            'nspin':int,
            'nstate':int
        }
        self.fp = None # h5py.File object (like a file pointer)
    def __del__(self):
      if self.fp is not None:
        self.fp.close()

    # =======================================================================
    # Basic Read Methods i.e. basic read/write and path access
    # =======================================================================
    def read(self,fname,force=False):
        """ open 'fname' for reading and save handle in this class """
        if not os.path.isfile(fname):
          raise RuntimeError('%s not found' % fname)
        if (self.fp is None) or force:
          self.fp = h5py.File(fname)
        else:
          raise RuntimeError('already tracking a file %s'%str(self.fp))

    def val(self,loc):
        """ get value array of an arbitrary entry at location 'loc' """
        return self.fp[loc].value

    def get(self,name):
        """ get value array of a known entry """
        loc   = self.locations[name]
        return self.fp[loc].value

    # =======================================================================
    # Advance Read Methods i.e. more specific to QMCPACK 3.0.0
    # =======================================================================

    # construct typical paths
    #  e.g. electrons/kpoint_0/spin_0/state_0
    @staticmethod
    def kpoint_path(ikpt):
        path = 'electrons/kpoint_%d' % (ikpt)
        return path
    @staticmethod
    def spin_path(ikpt,ispin):
        path = 'electrons/kpoint_%d/spin_%d' % (ikpt,ispin)
        return path
    @staticmethod
    def state_path(ikpt,ispin,istate):
        path = 'electrons/kpoint_%d/spin_%d/state_%d/' % (ikpt,ispin,istate)
        return path

    # access specific eigenvalue or eigenvector
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

    @classmethod
    def psig_to_psir(self,gvecs,psig,rgrid_shape,vol):
      """ contruct orbital given in planewave basis
       Inputs: 
        gvecs: gvectors in reciprocal lattice units i.e. integers
        psig: planewave coefficients, should have the same length as gvecs
        vol: simulation cell volume, used to normalized fft
       Output:
        rgrid: orbital on a real-space grid """
      assert len(gvecs) == len(psig)
    
      kgrid = np.zeros(rgrid_shape,dtype=complex)
      for igvec in range(len(gvecs)):
        kgrid[tuple(gvecs[igvec])] = psig[igvec]
      # end for
      rgrid = np.fft.ifftn(kgrid) * np.prod(rgrid_shape)/vol
      return rgrid
    # end def psig_to_psir

    def get_psir_from_psig(self,ikpt,ispin,istate,rgrid_shape=None,mesh_factor=1.0):
      """ FFT psig to psir at the given (kpoint,spin,state) """ 
      # get lattice which defines the FFT grid
      axes = self.get('axes')
      vol  = np.dot(np.cross(axes[0],axes[1]),axes[2])
      # get MO in plane-wave basis
      gvecs = self.get('gvectors').astype(int)
      psig_arr = self.psig(ikpt=ikpt,ispin=ispin,istate=istate)
      psig = psig_arr[:,0] + 1j*psig_arr[:,1]
      # determine real-space grid size (QMCPACK 3.0.0 convention)
      #  ref: QMCWaveFunctions/Experimental/EinsplineSetBuilder.cpp::ReadGvectors_ESHDF()
      if rgrid_shape is not None: # !!!! override grid size
        pass
      else:
        rgrid_shape = map(int, np.ceil(gvecs.max(axis=0)*4*mesh_factor) )
      # end if
      psir = self.psig_to_psir(gvecs,psig,rgrid_shape,vol)
      return psir
    # end def get_psir_from_psig

    # build entire eigensystem as a dataframe
    def eigensystem(self):
        """ construct dataframe containing eigenvalues and eigenvectors
         labeled by (kpoint,spin,state) indices """
        import pandas as pd

        data = []
        nkpt = self.get('nkpt')
        nspin= self.get('nspin')
        for ikpt in range(nkpt):
          k_grp = self.fp[self.kpoint_path(ikpt)]
          rkvec = k_grp['reduced_k'].value
          for ispin in range(nspin):
            spin_loc = self.spin_path(ikpt,ispin)
            sp_grp   = self.fp[spin_loc]
            nstate   = sp_grp['number_of_states'].value[0]
            evals    = sp_grp['eigenvalues'].value
            for istate in range(nstate):
              st_loc = self.state_path(ikpt,ispin,istate)
              st_grp = self.fp[st_loc]
              evector= st_grp['psi_g'].value # shape (ngvec,2) (real,complex)
              entry  = {'ikpt':ikpt,'ispin':ispin,'istate':istate,
                'reduced_k':rkvec,'evalue':evals[istate],'evector':evector}
              data.append(entry)
            # end for istate
          # end for ispin
        # end for ikpt
        df = pd.DataFrame(data).set_index(['ikpt','ispin','istate'],drop=True)
        return df
    # end def eigensystem

    # =======================================================================
    # Advance Write Methods, some specialized for pyscf
    # =======================================================================
    @staticmethod
    def create_electrons_group(h5_handle,gvec,df,nelec):
      """ create and fill the /electrons group in hdf5 handle
       Inputs:
         h5_handle: hdf5 handle generated by h5py.File
         gvec: 2D numpy array of reciprocal space vectors (npw,ndim)
         df: dataframe containing the eigensystem,
           indexed by (kpt,spin,state), contains (evalue,evector,reduced_k)
         nelec: a list of the number of electrons per atom (if no pseudopotential, then 'species_id' returned by system_from_cell should do)
       Output:
         None
       Effect:
         fill /electrons group in 'h5_handle' """
      flat_df = df.reset_index()
      kpoints = flat_df['ikpt'].unique()
      spins   = flat_df['ispin'].unique()
      nkpt,nspin = len(kpoints),len(spins)
      # transfer orbitals (electrons group)
      for ikpt in range(nkpt): 
        # !!!! assume no symmetry was used to generate the kpoints
        kpt_path = 'electrons/kpoint_%d'%ikpt
        kgrp = h5_handle.create_group(kpt_path)
        kgrp.create_dataset('num_sym',data=[1])
        kgrp.create_dataset('symgroup',data=[1])
        kgrp.create_dataset('weight',data=[1])

        rkvec = df.loc[ikpt,'reduced_k'].values[0]
        kgrp.create_dataset('reduced_k',data=rkvec)
        if ikpt == 0: # store gvectors in kpoint_0
            kgrp.create_dataset('gvectors',data=gvec)
            kgrp.create_dataset('number_of_gvectors',data=[len(gvec)])
        # end if 

        for ispin in range(nspin): # assume ispin==0
          nstate = len(df.loc[(ikpt,ispin)])
          spin_path = os.path.join(kpt_path,'spin_%d'%ispin)
          spgrp     = h5_handle.create_group(spin_path)
          spgrp.create_dataset('number_of_states',data=[nstate])
        
          evals = np.zeros(nstate) # fill eigenvalues during eigenvector read
          for istate in range(nstate):
              state_path = os.path.join(spin_path,'state_%d'%istate)
              psig = df.loc[(ikpt,ispin,istate),'evector']
              psig_path = os.path.join(state_path,'psi_g')
              h5_handle.create_dataset(psig_path,data=psig)
              evals[istate] = df.loc[(ikpt,ispin,istate),'evalue']
          # end for istate
          spgrp.create_dataset('eigenvalues',data=evals)
        # end for ispin
      # end for ikpt
      # transfer orbital info
      h5_handle.create_dataset('electrons/number_of_electrons',data=nelec)
      h5_handle.create_dataset('electrons/number_of_kpoints',data=[nkpt])
      # !!!! hard-code restricted orbitals
      h5_handle.create_dataset('electrons/number_of_spins',data=[1])
    # end def create_electrons_group

    @staticmethod
    def system_from_cell(h5_handle,cell,pseudized_charge=None):
      """ create and fill the /supercell and /atoms groups
       Inputs:
         h5_handle: hdf5 handle generated by h5py.File
         cell: pyscf.pbc.gto.Cell class
       Outputs:
         species_id: a list of atomic numbers for each atom
       Effect:
         fill /supercell and /atoms group in 'h5_handle'
      """

      # write lattice
      axes = cell.lattice_vectors() # always in bohr
      h5_handle.create_dataset('supercell/primitive_vectors',data=axes)


      # write atoms
      pos  = cell.atom_coords() # always in bohr
      elem = [cell.atom_symbol(i) for i in range(cell.natm)]
      assert len(pos) == len(elem)
      h5_handle.create_dataset('atoms/number_of_atoms',data=[len(elem)])
      h5_handle.create_dataset('atoms/positions',data=pos)

      # write species info
      species  = np.unique(elem)
      h5_handle.create_dataset('atoms/number_of_species',data=[len(species)])
      atomic_number = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6}
      number_of_electrons = {}
      species_map = {}
      for ispec,name in enumerate(species):
        species_map[name] = ispec
        spec_grp = h5_handle.create_group('atoms/species_%d'%ispec)

        # write name
        if name not in species_map.keys():
          raise NotImplementedError('unknown element %s' % name)
        # end if
        spec_grp.create_dataset('name',data=[name])

        # write atomic number and valence
        Zn   = atomic_number[name]
        spec_grp.create_dataset('atomic_number',data=[Zn])
        Zps  = Zn
        if pseudized_charge is None: # no pseudopotential, use bare charge
          pass
        else:
          Zps -= pseudized_charge[name]
        # end if
        number_of_electrons[name] = Zps
        spec_grp.create_dataset('valence_charge',data=[Zps])
      # end for ispec 
      species_ids = [species_map[name] for name in elem]
      h5_handle.create_dataset('atoms/species_ids',data=species_ids)
      nelec_list = [number_of_electrons[name] for name in elem]
      return nelec_list # return the number of valence electrons for each atom
    # end def system_from_cell

# end class PwscfH5
