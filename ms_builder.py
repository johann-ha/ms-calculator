# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:27:19 2015

@author: johannha
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton
# Calcul de Ms

# But first a function to read csv files.
def _read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = np.cast
    data = [[] for dummy in xrange(len(dtype))]
    fi = open(filename, 'r')
    # Skip the first line
    #fi.readline()
    for line in fi:
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(float(number))
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)

# A function to parse parameters tables
def _getParameters(filename, dtype, separator=','):
    
    cast = np.cast
    data = [[] for dummy in xrange(len(dtype))]
    fi = open(filename, 'r')
    # Skip the first line
    fi.readline()
    for line in fi:
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return np.rec.array(data, dtype=dtype)

def _compute_regression(AusCompo, W0FE):
    Wmui = 0
    W0i = 0
    Wmuj = 0
    W0j = 0
    Wmuk = 0
    W0k = 0
    Wmul = 0
    W0l = 0

    fdescr = np.dtype([('elt', 'S3'), ('Kmu', 'float32'), ('K0', 'float32')])
    filename = 'Ghosh_data.dat'
    Params = _getParameters(filename, fdescr)

    for el, conc in AusCompo.items():
        Kmu = Params['Kmu'][Params['elt']==el]
        K0 = Params['K0'][Params['elt']==el]
        if el in ['C', 'N']:
            Wmui = Wmui + (np.sqrt(conc) * Kmu) ** 2
            W0i = W0i + (np.sqrt(conc) * K0) ** 2
        elif el in ['CR', 'MN', 'MO', 'NB', 'SI', 'TI', 'V']:
            Wmuj = Wmuj + (np.sqrt(conc) * Kmu) ** 2
            W0j = W0j + (np.sqrt(conc) * K0) ** 2
        elif el in ['AL', 'CU', 'NI', 'W']:
            Wmuk = Wmuk + (np.sqrt(conc) * Kmu) ** 2
            W0k = W0k + (np.sqrt(conc) * K0) ** 2
        elif el in ['CO']:
            Wmul = Wmul + np.sqrt(conc) * Kmu
            W0l = W0l + np.sqrt(conc) * K0
    
    Wmu = np.sqrt(Wmui) + np.sqrt(Wmuj) + np.sqrt(Wmuk) + Wmul
    W0 = W0FE + np.sqrt(W0i) + np.sqrt(W0j) + np.sqrt(W0k) + W0l
    
    return Wmu, W0

class DG(object):
    fdescr = np.dtype([('T', 'float32'), ('DG', 'float32')])

    def __init__(self, dg_data_fname):
        DGN = _read_array(dg_data_fname, DG.fdescr, separator='  ')
        DGN.sort(order='T')
        self.thermodata = UnivariateSpline(DGN['T'], DGN['DG'], s=1)

    def __call__(self, T):
        return self.thermodata(T)

class CritDG(object):
    
    # A few constants (parameters) we need to work with
    K1 = 1010.
    p = 0.5 
    q = 1.5 
    Tmu = 300.
    W0FE = 836.

    def __init__(self, phase_compo):
        self.Wmu, self.W0 = _compute_regression(phase_compo, CritDG.W0FE)

    def __call__(self, T, DG):
        # DG has to be a DG object
        # need to defend against illegal calls?
        if T <= CritDG.Tmu:
            crit_DG = DG(T) + CritDG.K1 + self.Wmu + self.W0 * \
            (1.0 - (T / CritDG.Tmu) ** (1.0 / CritDG.q)) ** (1.0 / CritDG.p)
        elif T > CritDG.Tmu:
            crit_DG = DG(T) + CritDG.K1 + self.Wmu

        return crit_DG

class Ms(object):

    def __init__(self, dg_fname, phase_compo):
        self.thermo = DG(dg_fname)
        self.critDG = CritDG(phase_compo)

    def value(self):
        args = (self.thermo,)
        T0 = 500.
        Ms = newton(self.critDG, T0, args=args)
        return Ms.item()

if __name__ == '__main__':

    dg_fname = ['DGC_410_903K.dat', 'DGC_H078_843K.dat', 'DGC_H078_903K.dat']
    aus_compo = [{'MN' : 1.23011E-02, 'CU' : 1.28474E-03, 'CR' : 1.21947E-01,
                  'SI' : 9.71221E-03, 'N' : 1.12704E-04, 'NI' : 9.80304E-02,
                  'MO' :  2.84167E-03, 'C' : 3.06509E-05},
                 {'MN' : 4.38211E-02, 'MO' : 1.78767E-04, 'NI' : 1.38422E-01,
                  'SI' : 1.57016E-02, 'N' : 3.90219E-05, 'CR' : 1.19148E-01,  
                  'CU' : 2.70804E-04, 'C': 1.38558E-05},
                 {'MN' : 2.33411E-02, 'N' : 1.56561E-04, 'CR' : 1.29221E-01,
                  'SI' : 1.47015E-02, 'CU' : 1.38881E-04, 'NI' : 8.56657E-02,
                  'MO' : 2.00173E-04, 'C' : 7.68696E-05}]

    for fname, compo in zip(dg_fname, aus_compo):
        print fname[4:-4]
        mart_start = Ms(fname, compo)
        Ms_temperature = mart_start.value()
        print("Ms = {0}K, {1}C".format(Ms_temperature, Ms_temperature - 273.15))