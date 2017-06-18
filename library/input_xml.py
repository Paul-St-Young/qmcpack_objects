#import xml.etree.ElementTree as etree
import lxml.etree as etree
import numpy as np

class InputXml:
    def __init__(self):
        pass
    # end def
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

    # QMCPACK specific
    def find_pset(self,name='e'):
        """ return xml node specifying the particle set with given name
         by default return the quantum particle set 'e' """
        return self.find('.//particleset[@name="%s"]'%name)
    # end find_pset

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
            return self.text2arr(mytext)
          else:
            return np.array([line.split() for line in tlist],dtype=dtype) 
          # end if
        # end if
    # end def

    # =======================================================================
    # Advance Read Methods i.e. specific to pyscf or QMCPACK 3.0
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
    @staticmethod
    def atom_text(elem,pos):
      assert len(elem) == len(pos)
      lines = []
      for iatom in range(len(elem)):
          mypos = pos[iatom]
          line = '%5s  %10.6f  %10.6f  %10.6f' % (elem[iatom],mypos[0],mypos[1],mypos[2])
          lines.append(line)
      # end for iatom
      atext = ';\n'.join(lines)
      return atext
    # end def 
    @staticmethod
    def read_atoms(atext):
      # inverse of atom_text
      lines = atext.split(';')
      elem  = []
      pos   = []
      for line in lines:
          tokens = line.strip('\n').strip(' ').split()
          elem.append(tokens[0])
          pos.append( map(float,tokens[-3:]) )
      # end for line
      return np.array(elem),np.array(pos)
    # end def read_atoms
    """ # !!!dangerous!!!! hard to keep this consistent with pwscf.h5!
    def particleset_from_cell(self,cell,name='ion0'):
      elem,pos = self.read_atoms(cell.atom)
      species  = elem.unique()
      for name in species:
        atom_idx = np.where(elem==name)
        sp_grp = etree.Element('group',{'name':name,'size':len(atom_idx)})
      # end for name
      # build <particleset>
      pset_node = etree.Element('particleset',{'name':'ion0'})
      return pset_node
    # end def particleset_from_cell
    """
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

        # build xml node
        charge_node = etree.Element('parameter',{'name':'charge'})
        charge_node.text = str(atomic_number)
        valence_node = etree.Element('parameter',{'name':'valence'})
        valence_node.text = str(valence_charge)
        pos_node = etree.Element('attrib',{'name':'position','datatype':'posArray','condition':'0'}) # ? what is condition=0 ?
        pos_node.text = self.arr2text(pos_arr)
        grp_children = [charge_node,valence_node,pos_node]

        grp_node = etree.Element('group',{'name':name})
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
    # ----------------

# end class
