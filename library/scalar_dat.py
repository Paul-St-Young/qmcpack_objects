import numpy as np
import pandas as pd

class ScalarDat:
    def __init__(self):
        pass
    # end def __init__

    def read(self,fname):
        self.fname = fname
        self.data  = np.loadtxt(fname)
        with open(fname,'r') as f:
            header = f.readline().strip('#').split()
        # end with
        self.df = pd.DataFrame(self.data,columns=header)
    # end def read

# end class ScalarDat
