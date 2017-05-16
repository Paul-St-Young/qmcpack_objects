import xml.etree.ElementTree as ET

class InputXml:
    def __init__(self):
        pass
    # end def
    def read(self,fname):
        self.fname = fname
        self.root = ET.parse(fname)
    # end def
    def write(self,fname=None):
        if fname is None:
            self.root.write(self.fname)
        else:
            self.root.write(fname)
        # end if
    # end def
    
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
    # use str.split() for text2arr

# end class
