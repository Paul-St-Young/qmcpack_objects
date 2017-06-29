#!/usr/bin/env python
# extract crystal structure from QMCPACK input
import numpy as np

if __name__ == '__main__':
  
  qmcpack_input = 'opt.xml'
  
  # extract crystal structure from QMCPACK input
  from input_xml import InputXml
  inp = InputXml()
  inp.read(qmcpack_input)

  axes = inp.lattice_vectors()
  pos  = inp.atomic_coords()
  del inp

  natom,ndim = pos.shape

  # create reference structure
  import pymatgen as mg
  elem = ['H'] * natom
  struct = mg.Structure(axes,elem,pos,coords_are_cartesian=True)
  ref_upos   = struct.frac_coords
  ref_dtable = struct.distance_matrix

  # calculate fractional positions
  inv_axes = np.linalg.inv(axes)
  upos = np.dot(pos,inv_axes)
  assert np.allclose(ref_upos,upos)

  # calculate distance table
  dtable = np.zeros([natom,natom],float)
  from itertools import combinations,product
  # loop through all unique pairs of atoms
  for (i,j) in combinations(range(natom),2): # 2 for pairs
    dists = []
    images = product([-1,0,1],repeat=ndim) # check all neighboring images
    # loop through all neighboring periodic images of atom j
    #  should be 27 images for a 3D box
    for ushift in images:
      shift = np.dot(ushift,axes)
      disp  = pos[i] - (pos[j]+shift)
      dist  = np.linalg.norm(disp)
      dists.append(dist)
    # end for ushift
    dtable[i,j] = min(dists)
    dtable[j,i] = min(dists)
  # end for (i,j)
  assert np.allclose(dtable,ref_dtable)

# end if __main__
