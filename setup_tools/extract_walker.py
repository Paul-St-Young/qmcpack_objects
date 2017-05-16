#!/usr/bin/env python

import numpy as np
import sys
sys.path.insert(0,'../library')
import input_xml, h5_conf

def load_walker_as_pset(h5config,xml_inp,iwalker=0):
    """ load iwalker from h5config ('*.config.h5') and format into
     a dictionary suitable for xml_inp's quantum particle set
    """
    
    # get desired walker
    # ==========================================
    h5conf = h5_conf.H5Conf()
    h5conf.read(h5config)
    pos = h5conf.get_entry('walkers')[iwalker]
    if len(pos.shape) != 2:
        raise RuntimeError('wrong walker %s shape, expect (natom,ndim)',str(pos.shape))
    # end if
    
    inp = input_xml.InputXml()
    inp.read(xml_inp)
    # get simulation cell
    # ==========================================
    scell = inp.find('.//simulationcell')
    bconds_node = scell.find('.//parameter[@name="bconds"]')
    bconds = bconds_node.text.split()
    if bconds != ['p','p','p']:
        raise NotImplementedError('no support for open boundary or 2D yet')
    # end if
    axes_text = scell.find('.//parameter[@name="lattice"]').text
    axes = np.array(axes_text.split(),dtype=float).reshape(3,3)
    inv_axes = np.linalg.inv(axes)
    
    # put walker in simulation cell (-L/2,L/2)
    # ==========================================
    frac_pos = np.dot(pos,inv_axes)
    lower_bound = int( frac_pos.min()-1 )
    walker = np.dot((frac_pos %1)-0.5,axes)
    
    # get walker format from xml particleset
    # ==========================================
    tpset = inp.find_pset()
    groups = tpset.findall('.//group')
    
    # check that target particle set has the same number of particles as walker
    nptcls = map(int,[species.attrib['size'] for species in groups])
    ntot_xml = np.sum(nptcls)
    if ntot_xml != len(pos):
        raise RuntimeError('%d particles in xml, %d in walker' % (ntot_xml,len(pos)))
    # end if
    
    pset_dict = {}
    iptcl = 0
    for species in groups:
        name  = species.attrib['name']
        nptcl = int(species.attrib['size'])

        pos_text = inp.arr2text(walker[iptcl:iptcl+nptcl])
        pset_dict[name] = pos_text
        iptcl += nptcl
    # end for species
    
    return pset_dict
# end def load_walker_as_pset

if __name__ == '__main__':
    pass
