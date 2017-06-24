#import xml.etree.ElementTree as etree
import numpy as np
from copy import deepcopy
import lxml.etree as etree

class InputXml:

    def __init__(self):
        pass
    # end def

    # =======================================================================
    # Basic Methods (applicable to all xml files)
    # =======================================================================
    def read(self,fname):
        self.fname = fname
        parser = etree.XMLParser(remove_blank_text=True)
        self.root = etree.parse(fname,parser)
    # end def
    def write(self,fname=None,pretty_print=True):
        if fname is None:
            self.root.write(self.fname,pretty_print=pretty_print)
        else:
            self.root.write(fname,pretty_print=pretty_print)
        # end if
    # end def
    def show(self,node):
        """ print text representation of an xml node """
        print etree.tostring(node,pretty_print=True)
    
    # pass along xpath expression e.g. './/particleset'
    def find(self,xpath): 
        return self.root.find(xpath)
    # end def
    def find_all(self,xpath): 
        return self.root.findall(xpath)
    # end def

    @classmethod
    def arr2text(self,arr):
        """ format convert a numpy array into a text string """
        text = ''
        if len(arr.shape) == 1: # vector
            text = " ".join(arr.astype(str))
        elif len(arr.shape) == 2: # matrix
            mat  = [self.arr2text(line) for line in arr]
            text = "\n" + "\n".join(mat) + "\n"
        else:
            raise RuntimeError('arr2text can only convert vector or matrix.')
        # end if
        return text
    # end def

    @classmethod
    def text2arr(self,text,dtype=float,flatten=False):
        tlist = text.strip(' ').strip('\n').split('\n')
        if len(tlist) == 1:
          return np.array(tlist,dtype=dtype)
        else:
          if flatten:
            mytext = '\n'.join(['\n'.join(line.split()) for line in tlist])
            myarr = self.text2arr(mytext)
            return myarr.flatten()
          else:
            return np.array([line.split() for line in tlist],dtype=dtype) 
          # end if
        # end if
    # end def

    @classmethod
    def node2dict(self,node):
      entry = dict(node.attrib)
      if node.text:
        entry.update({'text':node.text})
      # end if
      return entry
    # end def node2dict

    # =======================================================================
    # Simple Methods Specific to QMCPACK
    # =======================================================================
    def find_pset(self,name='e'):
        """ return xml node specifying the particle set with given name
         by default return the quantum particle set 'e' """
        return self.find('.//particleset[@name="%s"]'%name)
    # end find_pset

    # =======================================================================
    # Advance Methods i.e. specific to pyscf or QMCPACK 3.0
    # =======================================================================

    # ----------------
    # simulationcell
    def simulationcell_from_cell(self,cell,bconds='p p p',lr_cut=15.0):
      """ construct the <simulationcell> xml element from pyscf.pbc.gto.Cell class
       Inputs:
         cell: pyscf.pbc.gto.Cell class, should have lattice_vectors() and unit
         bconds: boundary conditions in each of the x,y,z directions, p for periodic, n for non-periodic, default to 'p p p ' 
         lr_cut: long-range cutoff paramter rc*kc, default to 15
       Output: 
         etree.Element representing <simulationcell>
       Effect:
         none
      """

      # write primitive lattice vectors
      axes = cell.lattice_vectors() # rely on pyscf to return a.u.
      lat_node = etree.Element('parameter'
        ,attrib={'name':'lattice','units':'bohr'})
      lat_node.text = self.arr2text(axes)

      # write boundary conditions
      bconds_node = etree.Element('parameter',{'name':'bconds'})
      bconds_node.text = bconds

      # write long-range cutoff parameter
      lr_node = etree.Element('parameter',{'name':'LR_dim_cutoff'})
      lr_node.text = str(lr_cut)

      # build <simulationcell>
      sc_node = etree.Element('simulationcell')
      sc_node.append(lat_node)
      sc_node.append(bconds_node)
      sc_node.append(lr_node)
      return sc_node
    # end def simulationcell_from_cell
    # ----------------
      
    # ----------------
    # particleset
    ## !!!dangerous!!!! hard to keep this consistent with pwscf.h5!
    #def particleset_from_cell(self,cell,name='ion0'):
    #  elem = [cell.atomic_symbol(i) for i in range(cell.natm)]
    #  pos  = cell.atomic_coords() # reply on pyscf to return in a.u.
    #  assert len(pos) == len(elem)
    #  species  = elem.unique()
    #  for name in species:
    #    atom_idx = np.where(elem==name)
    #    sp_grp = etree.Element('group',{'name':name,'size':len(atom_idx)})
    #  # end for name
    #  # build <particleset>
    #  pset_node = etree.Element('particleset',{'name':'ion0'})
    #  return pset_node
    ## end def particleset_from_cell
    def particleset_from_hdf5(self,h5_handle):
      atom_grp = h5_handle.get('atoms')
      nspec = atom_grp.get('number_of_species').value[0]
      species_ids = atom_grp.get('species_ids').value
      positions = atom_grp.get('positions').value

      groups = []
      for ispec in range(nspec):
        # turn h5 group into dictionary (i.e. h5ls -d)
        sp_grp         = atom_grp.get('species_%d'%ispec)
        name           = sp_grp.get('name').value[0]
        valence_charge = sp_grp.get('valence_charge').value[0]
        atomic_number  = sp_grp.get('atomic_number').value[0]

        # locate particles of this species
        atom_idx = np.where(species_ids==ispec)
        pos_arr  = positions[atom_idx]
        natom    = len(pos_arr)

        # build xml node
        charge_node = etree.Element('parameter',{'name':'charge'})
        charge_node.text = str(valence_charge)
        valence_node = etree.Element('parameter',{'name':'valence'})
        valence_node.text = str(valence_charge)
        pos_node = etree.Element('attrib',{'name':'position','datatype':'posArray','condition':'0'}) # ? what is condition=0 ?
        pos_node.text = self.arr2text(pos_arr)
        grp_children = [charge_node,valence_node,pos_node]

        grp_node = etree.Element('group',{'name':name,'size':str(natom)})
        for child in grp_children:
          grp_node.append(child)
        # end for
        groups.append(grp_node)
      # end for ispec
      
      # build <particleset>
      pset_node = etree.Element('particleset',{'name':'ion0'})
      for group in groups:
        pset_node.append(group)
      # end for

      return pset_node
    # end def particleset_from_hdf5

    def ud_electrons(self,nup,ndown):
      up_group = etree.Element('group',{'name':'u','size':str(nup),'mass':'1.0'})
      dn_group = etree.Element('group',{'name':'d','size':str(ndown),'mass':'1.0'})
      for egroup in [up_group,dn_group]:
        charge_node = etree.Element('parameter',{'name':'charge'})
        charge_node.text = '  -1  '
        egroup.append( deepcopy(charge_node) )
      # end for egroup

      epset = etree.Element('particleset',{'name':'e','random':'yes'})
      epset.append(up_group)
      epset.append(dn_group)
      return epset
    # ----------------

    # ----------------
    # numerics
    # grid
    def radial_function(self,node):
      assert node.tag=='radfunc'

      # read grid definitions ( e.g. np.linspace(ri,rf,npts) for linear grid )
      gnode = node.find('.//grid')      # expected attributes: 
      grid_defs = self.node2dict(gnode) #   type,ri,rf,npts,units
      ri = float(grid_defs['ri'])
      rf = float(grid_defs['rf'])
      npts = int(grid_defs['npts'])
      gtype = grid_defs['type']
      units = grid_defs['units']

      # read 
      dnode = node.find('.//data')
      rval  = self.text2arr(dnode.text,flatten=True) # read as 1D vector
      assert len(rval) == npts
      entry = {'type':gtype,'units':units,'ri':ri,'rf':rf,'npts':npts,'rval':rval}
      return entry
    # end def radial_function
    # ----------------

# end class
