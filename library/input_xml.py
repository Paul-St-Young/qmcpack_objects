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
        print etree.tostring(node)
    
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
    def text2arr(self,text,dtype=float):
        tlist = text.strip(' ').strip('\n').split('\n')
        if len(tlist) == 1:
            return np.array(tlist,dtype=dtype)
        else:
            return np.array([line.split() for line in tlist],dtype=dtype) 
        # end if
    # end def

# end class
