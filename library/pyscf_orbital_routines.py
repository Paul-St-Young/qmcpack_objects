#!/usr/bin/env python
# assume periodic-boundary conditions i.e. import from pyscf.pbc
import numpy as np
import pandas as pd

def ao_on_grid(cell):
  from pyscf.pbc.dft import gen_grid,numint
  coords = gen_grid.gen_uniform_grids(cell)
  aoR    = numint.eval_ao(cell,coords)
  return aoR.astype(complex)
# end def ao_on_grid

def get_pyscf_psir(mo_vec,cell):
  """ 
  Inputs:
    mo_vec: 1D vector of AO coefficients representing a single MO
    cell: pyscf.pbc.gto.Cell object, used to exact AO on grid
  Output:
    moR: MO on grid
  """
  # get molecular orbital
  aoR = ao_on_grid(cell)
  rgrid_shape = np.array(cell.gs)*2+1 # shape of real-space grid
  assert np.prod(rgrid_shape) == aoR.shape[0]

  moR = np.dot(aoR,mo_vec)
  return moR.reshape(rgrid_shape)
# end def get_pyscf_psir

def mo_coeff_to_psig(mo_coeff,aoR,cell_gs,cell_vol,int_gvecs=None):
  """
   !!!! assume mo_coeff are already sorted from lowest to highest energy
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
    gvecs,eig_df = mo_orbitals(mf.mo_coeff,aoR,mf.cell.gs,mf.cell.vol)
    eig_df['ispin'] = ikpt
    eig_df['ikpt']  = ispin
    kpt_data = np.zeros([len(eig_df),len(mf.kpt)])
    kpt_data[:] = mf.kpt # copy kpt to every row of kpt_data
    eig_df['reduced_k'] = kpt_data.tolist()
    eig_df.set_index(['ikpt','ispin','istate'],inplace=True,
      drop=True)
    eig_df.sort_index()
    if save:
      eig_df.reset_index().to_json(eigsys_fname)
      np.savetxt(gvec_fname,gvecs)
    # end if
  # end if
  return gvecs,eig_df
# end def save_eigensystem

def mo_orbitals(mo_energy,mo_coeff,ao_on_grid,grid_shape,cell_volume):
  """ save molecular orbitals (MOs) in plane-wave basis in a database 
  Input:
    mo_energy: MO eigenvalues, written to database but only used to sort the MOs
    mo_coeff: MO eigenvectors in atomic orbital (AO) basis
    ao_on_grid: AOs on a real-space grid
    grid_shape: shape of the real-space grid 
    cell_volume: volume of the simulation cell, only used to normalized Fourier transform
  Output:
    gvecs: a list of integer vectors of shape (npw,ndim), which represent the plane-wave basis in reciprocal space units
    eig_df: a pandas dataframe, one entry for each MO
  """

  # perform Fourier transform to get plane-wave coefficients psig in basis gvecs
  gvecs,psig = mo_coeff_to_psig(mo_coeff,ao_on_grid,grid_shape,cell_volume)

  # build dataframe using sorted MOs
  sorted_state_indices = np.argsort(mo_energy)
  istate = 0
  data = []
  for idx in sorted_state_indices:
    entry = {'istate':istate,'evalue':mo_energy[idx],'evector':psig[idx,:,:]}
    data.append(entry)
    istate += 1
  # end for
  df = pd.DataFrame(data)
  return gvecs,df
# end def mo_orbitals

def multideterminant_coefficients(detlist,nfill,mo_coeff):
  """ save all mo_coefficients needed by a multideterminant expansion
  Inputs:
    detlist: list of determinants in MO basis, shape (ndet,nmo*nmo)
    nfill: number of filled orbitals for each determinant, integer 
    mo_coeff: MOs in AO basis, shape (nao,nmo)
  Outputs:
    new_mo_coeff: determinant orbitals in AO basis, shape (nao,ndet*nfill)
  """
  ndet,nb = detlist.shape
  nao,nmo = mo_coeff.shape
  assert nb == nao*nmo
  
  new_mo_coeff = np.zeros([nao,ndet*nfill],dtype=complex)

  for idet in range(ndet):
    for iorb in range(nfill):
      det = detlist[idet].reshape(nmo,nmo)
      new_mo_coeff[:,idet*nfill:idet*nfill+nfill] = np.dot(mo_coeff,det)[:,:nfill]
    # end for iorb
  # end for idet
  return new_mo_coeff
# end def multideterminant_coefficients

def save_multideterminant_orbitals(detlist,nfill,mf,ikpt=0,ispin=0):
  # !!!! obsolte, use multideterminant_orbitals instead
  aoR = ao_on_grid(mf.cell)
  new_mo_coeff = multideterminant_coefficients(detlist,nfill,mf.mo_coeff.astype(complex))
  norb = new_mo_coeff.shape[1]
  fake_mo_energy = np.arange(norb)

  fft_normalization = 1.0 # mf.cell.vol # !!!! do NOT use cell volume to normalize FFT (to match QMCPACK convention)
  gvecs,eig_df = mo_orbitals(fake_mo_energy,new_mo_coeff,aoR,mf.cell.gs,fft_normalization)

  # finish dataframe
  eig_df['ikpt']  = ikpt
  eig_df['ispin'] = ispin
  kpt_data = np.zeros([len(eig_df),len(mf.kpt)])
  kpt_data[:] = mf.kpt # copy kpt to every row of kpt_data
  eig_df['reduced_k'] = kpt_data.tolist()
  eig_df.set_index(['ikpt','ispin','istate'],inplace=True,drop=True)
  eig_df.sort_index()
  return gvecs,eig_df
# end def save_multideterminant_orbitals

def multideterminant_orbitals(detlist,nfill,cell,mo_coeff,kpt,ikpt=0,ispin=0):
  ndet,nmo0,nmo1 = detlist.shape

  aoR = ao_on_grid(cell) # get AOs on real-space grid
  new_mo_coeff = multideterminant_coefficients(detlist.reshape(ndet,nmo0*nmo1)
    ,nfill,mo_coeff.astype(complex))

  # energies of the orbitals are use to sort the orbitals only
  norb = new_mo_coeff.shape[1]
  fake_mo_energy = np.arange(norb)

  fft_normalization = 1.0 # cell.vol # !!!! do NOT use cell volume to normalize FFT (to match QMCPACK convention)
  gvecs,eig_df = mo_orbitals(fake_mo_energy,new_mo_coeff,aoR,cell.gs,fft_normalization)

  # finish dataframe
  eig_df['ikpt']  = ikpt
  eig_df['ispin'] = ispin
  kpt_data = np.zeros([len(eig_df),len(kpt)])
  kpt_data[:] = kpt # copy kpt to every row of kpt_data
  eig_df['reduced_k'] = kpt_data.tolist()
  eig_df.set_index(['ikpt','ispin','istate'],inplace=True,drop=True)
  eig_df.sort_index()
  return gvecs,eig_df
# end def multideterminant_orbitals

def uhf_multideterminant_spos(det_list,nfill_map,cell,mo_coeff,kpt):
  """
  Inputs:
    det_list: 4D numpy array of shape (ndet,nmo,nmo,nspin) 
    nfill_map: dict, map spin index to number of particles e.g. {0:4,1:4} for 4 up 4 down electrons
    cell: pyscf.pbc.gto.cell
    mo_coeff: 3D numpy array of shape (nspin,nao,nmo)
    kpt: 1D numpy array of shape (ndim,) 
  Outputs:
    gvecs: 2D numpy array of shape (npw,ndim)
    eig_df: Kohn-Sham orbitals in plane-wave basis 
  """
  ndet,nmo,nmo1,nspin = det_list.shape

  # reuse RHF routines
  gvecs_list  = []
  eig_df_list = []
  for ispin in range(nspin): # up and down
    print( 'det0 for spin %d is identity = %s'% (ispin,np.allclose(det_list[0,:,:,ispin],np.eye(nmo))) )
    gvecs, eig_df = multideterminant_orbitals(det_list[:,:,:,ispin]
      ,nfill_map[ispin],cell,mo_coeff[ispin],kpt,ikpt=0,ispin=ispin)
    if ispin > 0:
      assert np.allclose(gvecs,gvecs_list[0])
    else:
      gvecs_list.append(gvecs)
    # end if
    eig_df_list.append(eig_df)
  # end for ispin

  eig_df = pd.concat(eig_df_list)
  return gvecs,eig_df
# end def uhf_multideterminant_spos

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

def atom_text(elem,pos):
  """ convert elem,pos to text representation """
  assert len(elem) == len(pos)
  lines = []
  for iatom in range(len(elem)):
      mypos = pos[iatom]
      line = '%5s  %10.6f  %10.6f  %10.6f' % (elem[iatom],mypos[0],mypos[1],mypos[2])
      lines.append(line)
  atext = ';\n'.join(lines)
  return atext
# end def atom_text
