import numpy as np

def str2comp(line):
  tokens = line.strip(' ()\n').split(',')
  real,imag = map(float,tokens)
  return real+1j*imag
# end def str2comp

def val(line,sep='=',pos=-1):
  return line.split(sep)[pos].strip()
# end def val

def get_val(name,mm,sep='=',pos=-1):
  idx = mm.find(name)
  if idx == -1:
      raise RuntimeError('%s not found'%name)
  # end if
  mm.seek(idx)
  line = mm.readline()
  return val(line,sep,pos)
# end def get_val

def read_header(mm,header_loc=0):
  """ read the phfmol determinant file header 
  Inputs:
    mm: mmap object
    header_loc: int, any memory location before the header line
  Outputs:
    uhf: int, 0 for RHF, 1 for UHF, >1 for others
    nci: int, number of determinants
  Effect:
    mm will be rewund to header_loc
  """

  mm.seek(header_loc) # default is to rewind file

  # make sure this file is what I expect
  assert mm.find('FullMO') != -1
  assert mm.find('CMajor') != -1
  orb_type = get_val('TYPE',mm)
  assert orb_type == 'rotated'
  mm.seek(header_loc)
  det_format = get_val('FORMAT',mm)
  assert det_format == 'PHF'
  mm.seek(header_loc)
  uhf = int(get_val('UHF',mm))

  # get number of determinants
  mm.seek(header_loc)
  nci = int(get_val('NCI',mm))

  # rewind
  mm.seek(header_loc)

  return uhf,nci
# end def read_header

def read_next_det(mm,prefix='Determinant',max_nao=1024):
    """ Read the next Determinant in file pointed to by memory map mm
     read the determinant like a vector without any reshape
    Inputs:
      mm: mmap object
      max_nao: int, maximum number of atominc orbital in the determinant
    Outputs:
      det_idx: int, index of determinant
      det_data: 1D numpy array of size (nmo*nmo,), stores flattened determinant
    Effect:
      mm will point to the end of the parsed determinant
    """
    idx = mm.find(prefix)
    if idx == -1:
      return -1
    mm.seek(idx)

    det_data = []
    det_line = mm.readline()
    det_idx = int( det_line.split(':')[-1] )
    for i in range(max_nao*max_nao):
      cur_idx = mm.tell()
      line = mm.readline().strip()
      if line.startswith('(') and line.endswith(')'):
        pair = line.split(',')
        if len(pair) == 2:
          det_data.append(str2comp(line))
        elif len(pair) == 3:
          for num in line.split():
            det_data.append(str2comp(num))
          # end for
        else:
          raise RuntimeError('cannot handle %d numbers on a line'%len(pair))
        # end if
      else:
        mm.seek(cur_idx)
        break
      # end if
      if i >= (max_nao*max_nao-1):
        raise RuntimeError('may need to increase max_nao')
      # emd if
    # end for i

    return det_idx,np.array(det_data,dtype=complex)
# end def read_next_det

def parse_determinants(fname,nmo):
  """ read a list of determinants in the MO basis from file fname
  Inputs:
    fname: str, name of file that contains all the determinants 
    nmo: int, number of MOs, each determinant should be (nmo,nmo) 
  Outputs:
    det_list: 3D numpy array of shape (nspin,ndet,nmo,nmo), for RHF nspin=1, for UHF nspin=2
  Effect:
    mm will point to a location after all the determinants
  """

  from mmap import mmap
  with open(fname,'r+') as f:
    mm = mmap(f.fileno(),0)
  # end with

  uhf,ndet = read_header(mm)
  nspin = 1 # assume RHF
  if uhf: # if UHF, then two determinants per iteration
    nspin = 2
  # end if uhf

  det_list = np.zeros([ndet,nmo*nmo*nspin],dtype=complex)
  for idet in range(ndet):
    det_idx, mat_vec = read_next_det(mm)
    my_nmo_sq = len(mat_vec)
    if uhf:
      my_nmo_sq /= 2
    # end if
    my_nmo = int(round( np.sqrt(my_nmo_sq) ))
    if my_nmo != nmo:
      raise RuntimeError('wrong number of MOs, nmo=%d given in argument, but %d in determinant' % (nmo,my_nmo))
    # end if

    my_det = mat_vec.reshape(nmo,nmo,nspin,order='F') # phfmol outputs in column major, checked in read_header
    # spin index seems to be the last index, see find_uhf_index
    det_list[idet,:] = my_det.flatten()
  # end for

  return det_list
# end def parse_determinants

def arr2text(arr):
  """ convert an array of complex numbers to fortran style column of text """
  comp2str  = lambda num: '(%.16e,%.16e)'%(num.real,num.imag)
  text_list = [comp2str(num) for num in arr]
  return "\n".join(text_list)
# end def

def unparse_determinants(det_list_fname):
  """ inverse of parse_determinants """
  # read determinants
  det_list = np.loadtxt(det_list_fname).view(complex)
  ndet1,nmo2_ud = det_list.shape
  assert ndet1 == ndet
  nmo = int(round( np.sqrt(nmo2_ud/2) ))
  assert 2*nmo*nmo == nmo2_ud

  # !!!! hard code header for now
  header = '''  &FCI
  UHF = 1
  FullMO
  CMajor
  NCI=%d
  TYPE = rotated
  FORMAT = PHF
  /''' % ndet

  # fake CI coefficients
  ci_text = ''
  for ici in range(ndet):
    ci_text += "\n(%f,%f)" % (1.0,0.0)
  # end for

  print ndet,nmo

  det_text = ''
  for idet in range(ndet):
    det_text += "\n Determinant:%13d" % (idet+1) # 1-index in fortran
    det_ud = det_list[idet].reshape(nmo,nmo,nspin).flatten(order='F')
    det_text += "\n" + arr2text(det_ud)
  # end for idet

  return header + ci_text + det_text
# end def 

def read_phfrun_det_part(mm,mo_header_idx,nbas,real_or_imag):
  """ read either the real or the complex part of the determinant printed in phfrun.out
  Inputs: 
    mm: mmap of phfrun.out
    mo_header_idx: memory index of the beginning of ther determinant
    nbas: number of basis functions, which determines the shape of the determinant (nbas,nbas)
    real_or_imag: must be either 'real' or 'imag' 
  Outputs: 
    idet: index of the determinant read from the mo_header line
    mydet: either the real or the imaginary part of the determinant 
  """

  if (real_or_imag != 'real') and (real_or_imag != 'imag'):
    raise InputError('real_or_imag must be one of "real" or "imag"')
  # end if

  mydet = np.zeros([nbas,nbas])

  mm.seek(mo_header_idx)
  mo_header = mm.readline()
  ridx = mo_header.index(real_or_imag)
  assert mo_header[ridx:ridx+len(real_or_imag)] == real_or_imag

  col_line = mm.readline()
  col_idx  = np.array(col_line.split(),dtype=int) -1 # -1 to start index at 0

  nblock = int( np.ceil(float(nbas)/len(col_idx)) )
  first_block = True
  for iblock in range(nblock):
    if first_block: # already read col_idx
      first_block = False
    else:
      col_line = mm.readline()
      col_idx = np.array(col_line.split(),dtype=int) -1 # -1 to start index at 0
    # end if
    for ibas in range(nbas):
      row_line   = mm.readline()
      row_tokens = row_line.split()
      irow = int(row_tokens[0]) -1 # -1 to start index at 0
      row = np.array(row_tokens[1:],dtype=float)
      mydet[irow,col_idx] = row.copy()
    # end ibas
  # end iblock

  # check determinant for empty columns
  for icol in range(nbas):
    col  = mydet[:,icol]
    zvec = np.zeros(nbas)
    if np.allclose(col,zvec): # thank your past self if you get here
      raise RuntimeError('dev. error: empty column detected, either nbas or nblock is incorrect')
    # end if
  # end for 

  return mydet
# end def read_phfrun_det_part

def all_idx_with_label(mm,label,max_ndet):
  mm.seek(0)
  idx_list   = np.zeros(max_ndet,dtype=int)
  for idet in range(max_ndet):
    new_header_idx = mm.find(label)
    idx_list[idet] = new_header_idx
    if new_header_idx == -1:
      break
    elif idet == max_ndet-1:
      raise RuntimeError('increase max_ndet')
    # end if
    header_idx = new_header_idx
    mm.seek(header_idx)
    mm.readline()
  # end for idet
  return idx_list
# end def all_idx_with_label

def read_phfrun_det(mm,prefix,nbas,max_ndet=8192):
  idx_list = all_idx_with_label(mm,prefix,max_ndet)
  last_pos = np.where(idx_list==-1)[0][0]
  det_real = read_phfrun_det_part(mm,idx_list[last_pos-2],nbas,'real')
  det_imag = read_phfrun_det_part(mm,idx_list[last_pos-1],nbas,'imag')
  det = det_real + 1j*det_imag
  return det
# end def read_phfrun_det
