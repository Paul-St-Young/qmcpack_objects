#import xml.etree.ElementTree as etree
import numpy as np
from copy import deepcopy
import lxml.etree as etree

class InputXml:

    def __init__(self):
      # some things everyone should know
      self.atomic_number = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'Mn':25}
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
    def findall(self,xpath): 
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

    def find_two_body_jastrow(self,speciesA,speciesB):
      jas2 = self.find('.//jastrow[@type="Two-Body"]')
      xexpression = './/correlation[@speciesA="%s" and @speciesB="%s"]' % (speciesA,speciesB)
      results = jas2.xpath(xexpression)
      if len(results) == 0:
        return None
      elif len(results) == 1:
        return results[0]
      else:
        raise RuntimeError('conflicting definitions of Jastrow for '+speciesA+speciesB)
      # end if
    # end def find_two_body_jastrow

    def lattice_vectors(self):
      sc_node  = self.find('.//simulationcell')
      lat_node = sc_node.find('.//parameter[@name="lattice"]')
      unit = lat_node.get('units')
      assert unit == 'bohr'
      axes = self.text2arr( lat_node.text )
      return axes
    # end def lattice_vectors

    def atomic_coords(self,pset_name='ion0'):
      # !!!! assuming atomic units (bohr)
      source_pset_node = self.find_pset(pset_name)
      pos_node = source_pset_node.find('.//attrib[@name="position"]')
      pos = self.text2arr(pos_node.text)
      return pos
    # end def

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
    # end simulationcell
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
    # end def ud_electrons

    def group_positions(self,name):
      node = self.find('.//group[@name="%s"]'%name)
      pos_node = node.find('.//attrib[@name="position"]')
      pos_text = pos_node.text
      pos = self.text2arr(pos_text)

      # check
      nptcl = int( node.get('size') )
      assert len(pos) == nptcl

      return pos
    # end def group_positions
    # end particleset
    # ----------------

    # ----------------
    # hamiltonian

    # end hamiltonian
    # ----------------

    # ----------------
    # wavefunction

    def rhf_slater(self,wf_h5_fname,norb
      ,basis_type='whatever_spline',tilematrix='1 0 0 0 1 0 0 0 1',twistnum='0',source='ion0'
      ,meshfactor='1.0'   # set size of FFT grid, can be overwritten using fftgrid
      ,fftgrid=None       # manually set FFT grid shape, overwrites meshfactor, e.g. '16 16 16'
      ,precision='double' # use double precision for wavefunction evaluation
      ,truncate='no'      # 'no': do not truncate basis function for slab or open geometries
    ):
      """ write <sposet_builder> and <determinantset> into a <wavefunction> node
      Inputs:
        wf_h5_fname: str, location of hdf5 wavefunction file relative to QMCPACK run folder - file not read
        norb: int, number of orbitals
      Outputs:
        wf_node: etree.Element, <wavefunction> node
        **obsolete**sposet_builder_node: etree.Element, <sposet_builder> node
        **obsolete**determinantset_node: etree.Element, <determinantset> node
      """

      wf_node = etree.Element('wavefunction')
      sposet_builder_node = etree.Element('sposet_builder')
      determinantset_node = etree.Element('determinantset')
      slater_node = etree.Element('slaterdeterminant')
      determinantset_node.append(slater_node)

      wf_node.append(sposet_builder_node)
      wf_node.append(determinantset_node)

      # write <sposet_builder>
      spo_name = 'spo-ud' # !!!! hard code sposet name
      ispin    = 0        # !!!! hard code to use spindataset="0" for RHF
      name_val_pairs = zip(
        ['type','href','tilematrix','twistnum','source','meshfactor','precision','truncate'],
        [basis_type,wf_h5_fname,tilematrix,twistnum,source,meshfactor,precision,truncate]
      )
      for name,val in name_val_pairs:
        sposet_builder_node.set(name,val)
      # end for name

      if fftgrid is not None:
        sposet_builder_node.set('fftgrid',fftgrid)
      # end if

      spo_node = etree.Element('sposet',{'type':'bspline','name':spo_name,'size':str(norb),'spindataset':str(ispin)})
      sposet_builder_node.append(spo_node)

      # write <determinantset>, probably better to reference <particleset>
      for name,group in zip(['updet','downdet'],['u','d']):
        det_node = etree.Element('determinant',{'id':name,'group':group,'size':str(norb),'sposet':spo_name})
        slater_node.append(det_node)
      # end for

      return wf_node
    # end def rhf_slater

    def uhf_slater(self,wf_h5_fname,nptcl_map,spo_name_map={'u':'spo_up','d':'spo_dn'},spindataset_map={'u':0,'d':1}
      ,basis_type='whatever_spline',tilematrix='1 0 0 0 1 0 0 0 1',twistnum='0',source='ion0'
      ,meshfactor='1.0'   # set size of FFT grid, can be overwritten using fftgrid
      ,fftgrid=None       # manually set FFT grid shape, overwrites meshfactor, e.g. '16 16 16'
      ,precision='double' # use double precision for wavefunction evaluation
      ,truncate='no'      # 'no': do not truncate basis function for slab or open geometries
    ):
      """ write UHF wavefunction
      Input:
        wf_h5_fname: same as rhf_slater
        nptcl_map: a dictionary from str to int, map species name to of particles of that species
        spo_name_map: a dictionary from str to int, map species name to sposet name
        spindataset_map: a dictionary from str to int, map species name to spindataset index
      Output:
        wf_node: same as rhf_slater
      """
      assert len(nptcl_map) == len(spo_name_map)
      assert len(nptcl_map) == len(spindataset_map)
      from copy import deepcopy
      nptcl_dummy = 0
      wf_node = self.rhf_slater(wf_h5_fname,nptcl_dummy,basis_type,tilematrix,twistnum,source,meshfactor,fftgrid,precision,truncate)

      # get <sposet> node and empty out <sposet_builder>
      sposet_builder_node = wf_node.find('./sposet_builder')
      sposet_node = sposet_builder_node.find('./sposet')
      sposet_builder_node.remove(sposet_node)

      # rebuild <sposet_builder>
      for group in nptcl_map.keys():
        name  = spo_name_map[group]
        nptcl = nptcl_map[group]
        spindataset = spindataset_map[group]

        mysposet_node = deepcopy(sposet_node)
        mysposet_node.set('name',name)
        mysposet_node.set('spindataset',str(spindataset))
        mysposet_node.set('size',str(nptcl))
        sposet_builder_node.append(mysposet_node)
      # end for

      # re-link determinants
      for det_node in wf_node.findall('./determiant'):
        group = det_node.get('group')
        name  = spo_name_map[group]
        nptcl = nptcl_map[group]

        det_node.set('size',nptcl)
        det_node.set('sposet',name)
      # end for

      return wf_node
    # end def uhf_slater

    # end wavefunction
    # ----------------

    # ----------------
    # qmc driver

    def get_qmc_node(self,walkers,method='vmc',move='pbyp',checkpoint=0
      ,param_name_val_map = {
        'substeps':1,
        'steps':10,
        'timestep':1.0,
        'warmupsteps':10,
        'blocks':100,
        'usedrift':'yes'
       }):
      """ write optimization inputs
      Inputs:
        nsample: int, number of VMC samples used to calculate the hamiltonain and overlap matrices
        nloop: int, number of outer loops 
      Output:
        loop_node: etree.Element, <loop> node 
      """
      param_name_val_map.update({'walkers':walkers})

      # build <qmc> node
      vmc_node  = etree.Element('qmc',{'method':method,'move':move,'checkpoint':str(checkpoint)})
      for name,val in param_name_val_map.iteritems():
        param_node = etree.Element('parameter',{'name':name})
        param_node.text = str(val)
        vmc_node.append(param_node)
      # end for
      return vmc_node

    # end def get_qmc_node

    def get_optimization_node(self,nloop,method='linear',checkpoint=-1
      ,e_weight=0.95,urv_weight=0.0,rv_weight=0.05
      ,param_name_val_map = {
        'samples':16384,
        'substeps':10,
        'steps':1,
        'timestep':1.0,
        'warmupsteps':10,
        'blocks':100,
        'usedrift':'yes'
       }):
      """ write optimization inputs
      Inputs:
        nsample: int, number of VMC samples used to calculate the hamiltonain and overlap matrices
        nloop: int, number of outer loops 
      Output:
        loop_node: etree.Element, <loop> node
      """
       
      loop_node = etree.Element('loop',{'max':str(nloop)})
      nwalker = 1 # !!!! hard code one walker per MPI group, only 'samples' matter
      vmc_node = self.get_qmc_node(nwalker,method=method,checkpoint=checkpoint)
      loop_node.append(vmc_node)

      for cname,cval in zip(['energy','unreweightedvariance','reweightedvariance'],[e_weight,urv_weight,rv_weight]):
        cost_node = etree.Element('cost',{'name':cname})
        cost_node.text = str(cval)
        vmc_node.append(cost_node)
      # end for

      return loop_node
    # end def optimization

    # end qmc driver
    # ----------------

    # ----------------
    # numerics - mirror the put() method of each class

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

    def bspline_functor(self,node):
      assert node.tag=='correlation'

      # read bspline knot definitions 
      nsize = int(node.get('size'))
      try:
        rcut = float(node.get('rcut'))
      except:
        rcut = np.nan
      # end try
      try:
        cusp = float(node.get('cusp'))
      except:
        cusp = np.nan
      # end try

      # read knots
      coeff = node.find('.//coefficients')
      param_name = coeff.get('id')
      knots = np.array(coeff.text.split(),dtype=float)

      entry = {'size':nsize,'rcut':rcut,'cusp':cusp,'id':param_name,'coeff':knots}
      return entry
    # end def bspline_function
    # ----------------

# end class
