#!/usr/bin/env python
import numpy as np
from lxml import etree

def pos_in_box(pos,axes):
  """ return the position of particles in simulation cell axes """
  # put particles in box
  upos = np.dot(pos,np.linalg.inv(axes.T))
  sel = (upos<0) | (upos>1)
  upos[sel] = upos[sel] % 1
  new_pos = np.dot(upos,axes.T)
  return new_pos
# end def pos_in_box

def restart_conf_in_input(fconf,finp,fnew='next.xml',iconf=0):
  """ read one walker from config.h5 'fconf' and put electron positions into input xml 'finp', write new input to 'fnew'
  Inputs:
    fconf: str, name of config.h5 file
    finp: str, name of xml input
    fnew: str, name of new xml input, default 'next.xml'
    iconf: int, detault to 0, configuration index
  Output:
    None
  Effect:
    write fnew
  """
  from h5_conf import H5Conf
  from input_xml import InputXml

  # read electron positions
  obj = H5Conf()
  obj.read(fconf)
  pos = obj.get_entry('walkers')[iconf]
  nptcl,ndim = pos.shape

  # read old input file
  inp = InputXml()
  inp.read(finp)

  # put particles in simulation box (comment out for open system)
  axes = inp.lattice_vectors()
  pos = pos_in_box(pos,axes)

  # write electron positions into input
  iptcl = 0
  epset = inp.find_pset('e')
  epset.set('random','no')
  for group in epset.findall('group'): # should be ['u','d']
    nsize = int( group.get('size') )
    mypos = pos[iptcl:iptcl+nsize,:]

    pos_node = etree.Element('attrib',{'name':'position','datatype':'posArray'})
    pos_node.text = inp.arr2text(mypos)
    group.append(pos_node)
    
    iptcl += nsize
  # end for group
  assert iptcl == nptcl

  inp.write(fnew)
# end def restart_conf_in_input

if __name__ == '__main__':

  fconf = 'c2.s000.config.h5'
  finp  = 'vmc.xml'
  restart_conf_in_input(fconf,finp)

# end __name__
