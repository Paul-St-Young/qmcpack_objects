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
    def hamiltonian_interactions(self,epset_name,ipset_name,pseudos=None):

      ii_node = etree.Element('constant',{'type':'coulomb','name':'IonIon'
        ,'source':ipset_name,'target':ipset_name})
      ee_node = etree.Element('pairpot',{'type':'coulomb','name':'ElecElec'
        ,'source':epset_name,'target':epset_name})
      if pseudos is None:
        ei_node = etree.Element('pairpot',{'type':'coulomb','name':'ElecIon'
          ,'source':ipset_name,'target':epset_name})
      else:
        ei_node = etree.Element('pairpot',{'type':'pseudo','name':'PseudoPot'
          ,'source':ipset_name,'target':epset_name
          ,'wavefunction':'psi0','format':'xml'})
        for elem,psp in pseudos.iteritems():
          pseudo_node = etree.Element('pseudo',
            {'elementType':elem,'href':psp})
          ei_node.append(pseudo_node)
        # end for
      # end if

      ham_children = [ii_node,ee_node,ei_node]
      ham_node = etree.Element('hamiltonian'
        ,{'name':'h0','type':'generic','target':epset_name})
      for child in ham_children:
        ham_node.append(child)
      # end for
      return ham_node
    # end hamiltonian
    def sk(self):
      est_node = etree.Element('estimator',{'type':'sk','name':'sk','hdf5':'yes'})
      return est_node
    # end def sk
    def gr(self,num_bin):
      est_node = etree.Element('estimator',{'type':'gofr','name':'gofr','num_bin':str(num_bin)})
      return est_node
    # end def gr
    def spindensity(self,grid_shape):
      est_node = etree.Element('estimator',{'type':'spindensity','name':'spin_density','report':'yes'})
      param_node = etree.Element('parameter',{'name':'grid'})
      param_node.text = ' '.join(grid_shape.astype(str))
      est_node.append(param_node)
      return est_node
    # end def spindensity
    def sksp(self):
      est_node = etree.Element('estimator',{'type':'structurefactor','name':'sksp','report':'yes'})
      return est_node
    # end def sksp
    def localmoment(self,source='ion0'):
      est_node = etree.Element('estimator',{'type':'localmoment','name':'mloc','source':source})
      return est_node
    # end def localmoment

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
      spo_name = 'spo_ud' # !!!! hard code sposet name
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
      det_nodes =  wf_node.findall('.//determinant')
      if len(det_nodes) == 0:
        raise RuntimeError('found no determinant')
      # end if
      for det_node in det_nodes:
        group = det_node.get('group')
        name  = spo_name_map[group]
        nptcl = nptcl_map[group]

        det_node.set('size',str(nptcl))
        det_node.set('sposet',name)
      # end for

      return wf_node
    # end def uhf_slater

    # for multideterminant
    def occupy_str(self,istart,nfill,ntot):
      """ return strings like 11110000, 00001111, which represent occupation of single-particle states
      Inputs:
        istart: int, index of first filled orbital
        nfill: int, number of filled orbitals
        ntot: int, total number of orbitals in sposet
      Output:
        text: str, e.g. occupy(0,4,8) -> '11110000'
      """
      occ_arr = ['0'] * ntot
      occ_arr[istart:(istart+nfill)] = ['1'] * nfill
      text = ''.join(occ_arr)
      return text
    # end def occupy_str
    
    def multideterminant_from_ci(self,ci_coeff,nfill,nstate
      ,spo_up='spo_up',spo_dn='spo_dn'
      ,cutoff=1e-16,real_coeff=False):
      """ construct the <multideterminant> xml node from an array of CI coefficients
      Inputs:
        ci_coeff: list of float/complex, determinant expansion coefficients
        nfill: int, number of filled orbitals in each determinant 
        nstate: int, total number of orbitals in sposet
      Output:
        node: lxml.etree.Element, <multideterminant> node
      """
      print('obsolete function! Use uhf_multideterminant_from_ci instead')
      ndet = len(ci_coeff)
      node = etree.Element('multideterminant',{'optimize':'no','spo_up':spo_up,'spo_dn':spo_dn})
      detlist = etree.Element('detlist',{'size':str(ndet),'type':'DETS'
        ,'nca':'0','ncb':'0','nea':str(nfill),'neb':str(nfill)
        ,'cutoff':str(cutoff),'nstates':str(nstate)})
      node.append(detlist)
      for idet in range(ndet):
        if real_coeff:
          coeff_text = '%f' % ci_coeff[idet]
        else:
          coeff_text = '(%f,%f)' % (ci_coeff[idet].real,ci_coeff[idet].imag)
        # end if
        alpha = self.occupy_str(idet*nfill,nfill,nstate)
        beta = self.occupy_str(idet*nfill,nfill,nstate)
        det = etree.Element('ci',{'id':'CIcoeff_%d'%idet,'coeff':coeff_text,'alpha':alpha,'beta':beta})
        detlist.append(det)
      # end for idet
      return node
    # end def multideterminant_from_ci
    
    def uhf_multideterminant_from_ci(self,ci_coeff,nfill_map,nstate_map
      ,spo_name_map={0:'spo_up',1:'spo_dn'}
      ,cutoff=1e-16,real_coeff=False):
      """ construct the <multideterminant> xml node from an array of CI coefficients
      Inputs:
        ci_coeff:     list of float/complex, determinant expansion coefficients
        nfill_map:    dict int ispin -> int nfill, number of filled orbitals in each determinant for species
        nstate_map:   dict int ispin -> int nstate, total number of orbitals in sposet for species
        spo_name_map: dict int ispin -> str spo_name, name of sposet to assign to species
        cutoff: float, ignore CI coefficients below cutoff
        real_coeff: write CI coefficients as real numbers to use real code
      Output:
        node: lxml.etree.Element, <multideterminant> node
      """
      up_dn_msg = 'multideterminant in QMCPACK is hard-coded for up and down electrons for now'
      assert len(spo_name_map) == 2, up_dn_msg
      assert len(nfill_map)    == 2, up_dn_msg
      assert len(nstate_map)   == 2, up_dn_msg

      nstate_up = nstate_map[0]; nstate_dn = nstate_map[1]
      assert nstate_up == nstate_dn, 'number of available orbitals for up and down electrons must be equal by hard code in QMCPACK'
      spo_up = spo_name_map[0]; spo_dn = spo_name_map[1]
      nfill_up = nfill_map[0]; nfill_dn = nfill_map[1]

      # initialize <multideterminant> <detlist/> </multideterminant>
      ndet = len(ci_coeff) 
      node = etree.Element('multideterminant',{'optimize':'no','spo_up':spo_up,'spo_dn':spo_dn})
      detlist = etree.Element('detlist',{'size':str(ndet),'type':'DETS'
        ,'nca':'0','ncb':'0','nea':str(nfill_up),'neb':str(nfill_dn)
        ,'cutoff':str(cutoff),'nstates':str(nstate_up)})
      node.append(detlist)

      # fill <detlist>
      for idet in range(ndet):
        if real_coeff:
          coeff_text = '%f' % ci_coeff[idet]
        else:
          coeff_text = '(%f,%f)' % (ci_coeff[idet].real,ci_coeff[idet].imag)
        # end if
        alpha = self.occupy_str(idet*nfill_up,nfill_up,nstate_up)
        beta = self.occupy_str(idet*nfill_dn,nfill_dn,nstate_dn)
        det = etree.Element('ci',{'id':'CIcoeff_%d'%idet,'coeff':coeff_text,'alpha':alpha,'beta':beta})
        detlist.append(det)
      # end for idet
      return node
    # end def multideterminant_from_ci

    def uhf_multidet_qmc(self,ci_coeff,nup,ndn,fftgrid,h5_href='pyscf2qmcpack.h5',real_coeff=False):
      ndet       = len(ci_coeff)
      wf_node = self.uhf_slater(h5_href,{'u':nup*ndet,'d':ndn*ndet},fftgrid=' '.join(fftgrid.astype(str)))
      mdet_node = self.uhf_multideterminant_from_ci(ci_coeff,{0:nup,1:ndn},{0:nup*ndet,1:ndn*ndet},real_coeff=real_coeff)
      # use default spo_name_map for now, !!!! check consistency with <sposet_builder>
    
      # swap out <slaterdeterminant> for <multideterminant>
      ds_node = wf_node.find('.//determinantset')
      sd_node = ds_node.find('.//slaterdeterminant')
      ds_node.remove(sd_node)
      ds_node.append(mdet_node)
      return wf_node
    # end def uhf_multidet_qmc

    def one_body_jastrow(self,ipset_node,cusp=None):
      ion_pset_name = ipset_node.get('name')
      jas_node  = etree.Element('jastrow',{'type':'One-Body','name':'J1','function':'bspline','source':ion_pset_name})
      for group in ipset_node.findall('group'):
        ion_name = group.get('name')
        corr_node = self.bspline_functor_from_dict()
        corr_node.set('elementType',ion_name)
        if cusp is not None:
          corr_node.set('cusp',cusp)
        # end if
        coeff_node = corr_node.find('.//coefficients')
        coeff_node.set('id','e%s'%ion_name)
        jas_node.append(corr_node)
      # end for
      return jas_node
    # end def one_body_jastrow

    def two_body_jastrow(self,epset_node):
      jas_node = etree.Element('jastrow',{'type':'Two-Body','name':'J2','function':'bspline'})
      species_names = [group.get('name') for group in epset_node.findall('.//group')]
      assert species_names == ['u','d'] # !!!! hard code for up and down electrons for now
      spec1 = 'u'
      for spec2 in ['u','d']:
        corr_node = self.bspline_functor_from_dict()
        corr_node.set('speciesA',spec1)
        corr_node.set('speciesB',spec2)
        coeff_node = corr_node.find('.//coefficients')
        coeff_node.set('id',spec1+spec2)
        jas_node.append(corr_node)
      # end for
      return jas_node
    # end def two_body_jastrow


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
      qmc_node  = etree.Element('qmc',{'method':method,'move':move,'checkpoint':str(checkpoint)})
      for name,val in param_name_val_map.iteritems():
        param_node = etree.Element('parameter',{'name':name})
        param_node.text = str(val)
        qmc_node.append(param_node)
      # end for
      return qmc_node

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
      vmc_node = self.get_qmc_node(nwalker,method=method,checkpoint=checkpoint
        ,param_name_val_map = param_name_val_map)
      loop_node.append(vmc_node)

      for cname,cval in zip(['energy','unreweightedvariance','reweightedvariance'],[e_weight,urv_weight,rv_weight]):
        cost_node = etree.Element('cost',{'name':cname})
        cost_node.text = str(cval)
        vmc_node.append(cost_node)
      # end for

      return loop_node
    # end def get_optimization_node

    def get_dmc_nodes(self,target_walkers,time_step_list=[0.02,0.01]
      ,correlation_time = 1.0
      ,nvmc_walkers     = 1
      ,checkpoint       = 0
      ,param_name_val_map = {
        'substeps':1,
        'steps':10,
        'timestep':1.0,
        'warmupsteps':10,
        'blocks':40,
        'usedrift':'yes'
       }):

      vmc_nv_map = param_name_val_map.copy()
      vmc_nv_map.update({'samples':target_walkers})
      vmc_node = self.get_qmc_node(nvmc_walkers,param_name_val_map=vmc_nv_map,checkpoint=checkpoint)

      nodes = [vmc_node] # use a VMC to get initial walkers
      for ts in time_step_list:
        dmc_nv_map = param_name_val_map.copy()

        steps = int(round( float(correlation_time)/ts ))
        if steps < 1:
          raise RuntimeError('decrease time step or increase correlation_time')
        # end if
        dmc_nv_map['timestep'] = str(ts)
        dmc_nv_map['steps'] = str(steps)

        dmc_nv_map.update({'targetwalkers':target_walkers})
        dmc_node = self.get_qmc_node(1,method='dmc',param_name_val_map=dmc_nv_map,checkpoint=checkpoint)
        nodes.append(dmc_node)
      # end for ts

      return nodes
    # end get_dmc_node

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
    # end def bspline_functor

    def bspline_functor_from_dict(self,entry={
        'size':8,
        'rcut':None,
        'cusp':None,
        'id':'coeff',
        'optimize':'yes',
        'coeff':np.array([0,0,0,0,0,0,0,0])
        }):
      """ create <correlation> node readable by BsplineFunctor from a dictionary
      Input:
        entry: dict, must have ['size','rcut','cusp','id','optimize']
      Output:
        node: etree.Element, <correlation>
      """
      node = etree.Element('correlation')
      for key in ['size','rcut','cusp']:
        val = entry[key]
        if val is not None:
          node.set(key,str(val))
        # end if
      # end for

      coeff = etree.Element('coefficients')
      for key in ['id','optimize']:
        val = entry[key]
        if val is not None:
          coeff.set(key,str(val))
        # end if
      # end for
      coeff.set('type','Array')
      coeff.text = self.arr2text(entry['coeff'])
      node.append(coeff)

      return node
    # end def bspline_functor_from_dict

    # ----------------
    
    def write_qmcpack_input(self,inp_name,cell,wf_h5_fname,nup,ndn,wf_node=None,pseudos=None,qmc_nodes=[],proj_id='qmc'):
      import h5py
      from lxml import etree
    
      # build <project>
      proj_node = etree.Element('project',{'id':proj_id,'series':'0'})
    
      # build <simulationcell>
      sc_node   = self.simulationcell_from_cell(cell)
    
      # build <particleset>
      elec_pset_node= self.ud_electrons(nup,ndn)
      fp = h5py.File(wf_h5_fname)
      ion_pset_node = self.particleset_from_hdf5(fp)
    
      # build <wavefunction>
      #  in another file
    
      # build <hamiltonian>
      ion_pset_name = ion_pset_node.get('name')
      elec_pset_name= elec_pset_node.get('name')
      ham_node = self.hamiltonian_interactions(elec_pset_name,ion_pset_name,pseudos=pseudos)
    
      # assemble <qmcsystem>
      sys_node = etree.Element('qmcsystem')
      sys_children = [proj_node,sc_node,elec_pset_node,ion_pset_node,ham_node]
    
      for child in sys_children:
        # if give, insert <wavefunction> before <hamiltonian> 
        if (child.tag == 'hamiltonian') and (wf_node is not None):
          sys_node.append(wf_node)
        # end if
        sys_node.append(child)
      # end for
    
      # write input
      root = etree.Element('simulation')
      doc = etree.ElementTree(root)
      root.append(sys_node)
    
      # take <qmc> block from else where
      if len(qmc_nodes) > 0:
        for qmc_node in qmc_nodes:
          root.append(qmc_node)
        # end for
      # end if
    
      doc.write(inp_name,pretty_print=True)
    # end def write_qmcpack_input

# end class
