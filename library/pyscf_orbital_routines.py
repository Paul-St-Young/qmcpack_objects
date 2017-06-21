#!/usr/bin/env python
# assume periodic-boundary conditions i.e. import from pyscf.pbc
import numpy as np

def ao_on_grid(cell):
  from pyscf.pbc.dft import gen_grid,numint
  coords = gen_grid.gen_uniform_grids(cell)
  aoR    = numint.eval_ao(cell,coords)
  return aoR
# end def ao_on_grid

def mo_coeff_to_psig(mo_coeff,aoR,cell_gs,cell_vol,int_gvecs=None):
  """
   Inputs:
     mo_coeff: molecular orbital in AO basis, each column is an MO, shape (nao,nmo)
     aoR: atomic orbitals on a real-space grid, each column is an AO, shape (ngrid,nao)
     cell_gs: 2*cell_gs+1 should be the shape of real-space grid (e.g. (5,5,5))
     cell_vol: cell volume, used for FFT normalization
     int_gvecs: specify the order of plane-waves using reciprocal lattice points
   Outputs:
       3. plane-wave coefficients representing the MOs, shape (ngrid,nmo)
  """
  # provide the order of reciprocal lattice vectors to skip
  if int_gvecs is None: # use internal order
    nx,ny,nz = cell_gs
    from itertools import product
    int_gvecs = np.array([gvec for gvec in product(
      range(-nx,nx+1),range(-ny,ny+1),range(-nz,nz+1))],dtype=int)
  else:
    assert (int_gvecs.dtype is int)
  # end if
  npw = len(int_gvecs) # number of plane waves 

  # put molecular orbitals on real-space grid
  moR = np.dot(aoR,mo_coeff)
  nao,nmo = moR.shape
  rgrid_shape = 2*np.array(cell_gs)+1
  assert nao == np.prod(rgrid_shape)

  # for each MO, FFT to get psig
  psig = np.zeros([nmo,npw,2]) # store real & complex
  for istate in range(nmo):
    # fill real-space FFT grid
    rgrid = moR[:,istate].reshape(rgrid_shape)
    # get plane-wave coefficients (on reciprocal-space FFT grid)
    moG   = np.fft.fftn(rgrid)/np.prod(rgrid_shape)*cell_vol
    # transfer plane-wave coefficients to psig in specified order
    for igvec in range(npw):
      comp_val = moG[tuple(int_gvecs[igvec])]
      psig[istate,igvec,:] = comp_val.real,comp_val.imag
    # end for igvec
  # end for istate
  return int_gvecs,psig
# end def mo_coeff_to_psig

def save_eigensystem(mf,gvec_fname = 'gvectors.dat'
   ,eigsys_fname = 'eigensystem.json',save=True):
  import os
  import pandas as pd
  if os.path.isfile(eigsys_fname) and os.path.isfile(gvec_fname):
    gvecs = np.loadtxt(gvec_fname)
    eig_df = pd.read_json(eigsys_fname).set_index(
      ['ikpt','ispin','istate'],drop=True).sort_index()
  else:
    data = []
    ikpt  = 0 # gamma-point calculation
    ispin = 0 # restricted (same orbitals for up and down electrons)
    # get MOs in plane-wave basis
    aoR = ao_on_grid(mf.cell)
    gvecs,psig = mo_coeff_to_psig(mf.mo_coeff,aoR,mf.cell.gs,mf.cell.vol)
    nstate,npw,ncomp = psig.shape
    for istate in range(nstate):
      entry = {'ikpt':ikpt,'ispin':ispin,'istate':istate,
        'reduced_k':mf.kpt,'evalue':mf.mo_energy[istate],'evector':psig[istate,:,:]}
      data.append(entry)
    # end for istate
    eig_df = pd.DataFrame(data).set_index(
      ['ikpt','ispin','istate'],drop=True).sort_index()
  # end if
  if save:
    eig_df.reset_index().to_json(eigsys_fname)
    np.savetxt(gvec_fname,gvecs)
  # end if
  return gvecs,eig_df
# end def save_eigensystem

def generate_pwscf_h5(cell,gvecs,eig_df,pseudized_charge=None,h5_fname='pyscf2pwscf.h5'):
  
  # if eigensystem was saved to disk, use the following to read
  #import numpy as np
  #import pandas as pd
  #gvecs = np.loadtxt('../1_eigsys/gvectors.dat')
  #eig_df= pd.read_json('../1_eigsys/eigensystem.json').set_index(
  #  ['ikpt','ispin','istate'],drop=True).sort_index()

  # e.g. pseudized_charge = {'C':2}
  
  import h5py
  from pwscf_h5 import PwscfH5
  new = h5py.File(h5_fname,'w')
  ref = PwscfH5()
  nelecs = ref.system_from_cell(new,cell,pseudized_charge=pseudized_charge)
  ref.create_electrons_group(new,gvecs,eig_df,nelecs)
  
  # transfer version info. !!!! hard code for now
  new.create_dataset('application/code',data=['pyscf'])
  new.create_dataset('application/version',data=['1.4a'])
  new.create_dataset('format',data=['ES-HDF'])
  new.create_dataset('version',data=[2,1,0])
  new.close()
# end def generate_pwscf_h5
