from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

from io import open
import six
import textwrap

import opacplot2 as opp
import opacplot2.utils

import numpy as np
from .constants import KELVIN_TO_EV, GPA_TO_ERGCC, MJKG_TO_ERGCC, ERGCC_TO_GPA, ERGG_TO_MJKG

import periodictable as ptab

class OpgSesame:
    """
    This class is responsible for loading all SESAME formatted data files.

    ``OpgSesame`` reads in a SESAME file. Each key:value pair of the ``data``
    attribute corresponds to a table ID and its data from the file, respectively.

    Parameters
    ----------
    fn : str
       Name of file to open.

    precision : int
       ``opacplot2.OpgSesame.SINGLE`` for entry lengths of 15
       or ``opacplot2.OpgSesame.Double`` for entry lengths of 22.

    verbose : bool
       Verbose option.

    Attributes
    ----------
    data : dict
        Dictionary of material IDs included in the SESAME file.

    Examples
    --------
    The ``opacplot2.OpgSesame.data`` dictionary will
    hold the EoS data for the table referenced by ``table_id``. For example, if
    we are in a directory with the file ``sesame.ses``::

       >>> import opacplot2 as opp
       >>> op = opp.OpgSesame('sesame.ses', opp.OpgSesame.SINGLE)
       >>> print(op.data.keys())
       dict_keys([..., 13719]) # Table ID numbers; Aluminum.
       >>> data = op.data[13719]
       >>> print(sorted(data.keys()))
       dict_keys(['abar',...,'zmax']) # Dictionary containing EoS data.
    """

    SINGLE = 1
    DOUBLE = 2

    CHAR_LINE_LEN = 80
    WORDS_PER_LINE = 5

    def __init__(self, filename, precision, verbose=False):
        self.verbose = verbose
        self.fhand = open(filename, encoding='utf-8')
        if(precision == self.SINGLE):
           self.entry_len = 15
        elif(precision == self.DOUBLE):
            self.entry_len = 22
        else:
            raise ValueError("precision must be SINGLE or DOUBLE")


        self.fdict = { 101 : self.parseComment,
                       102 : self.parseComment,
                       103 : self.parseComment,
                       104 : self.parseComment,
                       201 : self.parseInfo,
                       301 : self.parseEos,
                       303 : self.parseEos,
                       304 : self.parseEos,
                       305 : self.parseEos,
                       306 : self.parseEos,
                       311 : self.parseEos,
                       321 : self.parseEos,
                       401 : self.parseVap,
                       411 : self.parseSolid,
                       412 : self.parseLiquid,
                       431 : self.parseShear,
                       432 : self.parseShear,
                       504 : self.parseZbar,
                       601 : self.parseZbar,
                       602 : self.parseEcond,
                       603 : self.parseTcond,
                       604 : self.parseTcond,
                       605 : self.parseTcond,
                       }

        self.data = {}

        self.recs = {}

        self.parse()


    def parse(self):

        while True:
            header = self.fhand.readline()
            print (header.split())
            if not header: break # Reached EOF
            if header[:3] == " 2 ": break

            matid = int(header.split()[1])
            recid = int(header.split()[2])
            nentries = int(header.split()[3])

            if not matid in self.data: self.data[matid] = {}

            if self.verbose and (recid > 104):
                print("Material = %8i  Record = %8i  Entries = %8i" % (matid, recid, nentries))

            # If there are opacity tables (series 500),
            # we do not currently read them.
            if (600>recid) and (recid>=500):
                print("Warning: No reader for SESAME opacity tables. The "
                      "data in table number {} will not be parsed".format(recid))


            if not recid in self.fdict:
                raise ValueError("No handling function for record %d" % recid)


            self.fdict[recid](nentries,matid, recid)

            if matid not in self.recs.keys():
                self.recs[matid] = [recid]
            else:
                self.recs[matid] = self.recs[matid] + [recid]

    def parseComment(self, nentries, matid, recid):

        nlines = (nentries-1) // self.CHAR_LINE_LEN + 1
        nchar = nentries + nlines
        return self.fhand.read(nchar)

    def parseInfo(self, nentries, matid, recid):
        words = self.readEntries(nentries)
        self.data[matid]["zmax"] = words[0]
        self.data[matid]["abar"] = words[1]
        self.data[matid]["rho0"] = words[2]
        self.data[matid]["bulkmod"] = words[3]
        self.data[matid]["excoef"] = words[4]

        if self.verbose: print("  zbar = %g  abar = %g  rho0 = %g" % (words[0],words[1],words[2]))

        return self.data

    def parseEos(self, nentries, matid, recid):
        words = self.readEntries(nentries)

        prefixes = { 301 : "total_",
                     303 : "ioncc_",
                     304 : "ele_",
                     305 : "ion_",
                     306 : "cc_" }

        prefix = prefixes[recid]

        ndens = int(words[0])
        ntemp = int(words[1])
        start = 2

        self.data[matid][prefix+"ndens"] = ndens
        self.data[matid][prefix+"ntemp"] = ntemp

        self.data[matid][prefix+"dens"] = words[start:start+ndens] # In g/cc
        start = start+ndens

        self.data[matid][prefix+"temps"] = words[start:start+ntemp]*KELVIN_TO_EV
        start = start + ntemp

        if self.verbose and prefix == "total_":
            dens = self.data[matid]["total_dens"]
            temps = self.data[matid]["total_temps"]

            print("ndens   = %13i ntemp    = %13i" % (ndens, ntemp))
            print("dens[0] = %13.6e  dens[-1] = %13.6e" % (dens[0], dens[-1]))
            print("temp[0] = %13.6e  temp[-1] = %13.6e" % (temps[0], temps[-1]))


        # Read pressure array (in GPa and convert to ergs/cc):
        self.data[matid][prefix+"pres"] = words[start:start+ntemp*ndens].reshape((ntemp,ndens)).transpose()*GPA_TO_ERGCC
        start = start + ntemp*ndens

        # Read specific internal energy array (in MJ/kg and convert to ergs/g):
        self.data[matid][prefix+"eint"] = words[start:start+ntemp*ndens].reshape((ntemp,ndens)).transpose()*MJKG_TO_ERGCC
        start = start + ntemp*ndens

        if start == nentries:
            # There is no Helmholtz free energy data, this is the end
            # of this record...
            return

        # Read specific Helmholtz free energy array (in MJ/kg and convert to ergs/g):
        self.data[matid][prefix+"free"] = words[start:start+ntemp*ndens].reshape((ntemp,ndens)).transpose()*MJKG_TO_ERGCC
        start = start + ntemp*ndens

        # exit()

    def parseVap(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def parseSolid(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def parseLiquid(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def parseShear(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def parseZbar(self, nentries, matid, recid):
        words = self.readEntries(nentries)

        prefix = "zbar"

        ndens = int(words[0])
        ntemp = int(words[1])
        start = 2

        self.data[matid][prefix+"_ndens"] = ndens
        self.data[matid][prefix+"_ntemp"] = ntemp

        self.data[matid][prefix+"_dens"] = 10**words[start:start+ndens]
        start = start+ndens

        self.data[matid][prefix+"_temps"] = 10**words[start:start+ntemp]
        start = start + ntemp

        self.data[matid][prefix] = 10**(words[start:start+ntemp*ndens].reshape((ntemp,ndens)).transpose())

    def parseEcond(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def parseTcond(self, nentries, matid, recid):
        words = self.readEntries(nentries)

    def readEntries(self,nentries):
        nlines = (nentries-1) // self.WORDS_PER_LINE + 1

        data = np.empty(nentries)

        string = ""
        for i in range(nlines):
            string += self.fhand.readline()[:self.WORDS_PER_LINE*self.entry_len]

        for i in range(nentries):
            word = string[self.entry_len*i:self.entry_len*(i+1)]
            if word[-4] == '-': word = word[:-4] + "E" + word[-4:]
            data[i] = float(word)

        return data

    def toEosDict(self, Znum=None, Anum=None,
                  Xnum=None, qeos=False, log=None,
                  filter_dens=0., filter_temps=0.,
                  tabnum=None):
        # For SESAME, we need the hedp package to calculate zbar.
        try:
            from hedp import eos
        except ImportError:
            raise ImportError('You need the hedp module. You can get it here: '
                              'https://github.com/luli/hedp.')

        if tabnum is None:
            # Select the last table (newest) table available.
            opp_ses_data = self.data[sorted(self.data.keys())[-1]]
        else:
            try:
                opp_ses_data = self.data[tabnum]
            except KeyError:
                raise KeyError('Invalid table number!')

        # Sesame has extra data points in it, so we must merge them down. We are
        # merging the default grids ioncc_ and ele_.
        # Information about filtering dens/temp grids is available here:
        # http://flash.uchicago.edu/pipermail/flash-users/2015-April/001689.html
        # We must filter dens > 0. in order to avoid problems with calculating
        # zbar below, since eos.thomas_fermi_ionization() returns `nan` where
        # density is 0.
        if not qeos:
            opp_ses_data = opp.utils.EosMergeGrids(
                                        opp_ses_data,
                                        filter_dens=lambda x: (x>filter_dens),
                                        filter_temps=lambda x: (x>filter_temps))

        # If we are dealing with SESAME generated by qeos, we need to merge
        # ion_ and ele_ grids rather than ioncc_ and ele_. We use the same
        # filters as above.
        if qeos:
            opp_ses_data = opp.utils.EosMergeGrids(
                                opp_ses_data, intersect=['ele', 'ion'],
                                filter_dens=lambda x: (x>filter_dens),
                                filter_temps=lambda x: (x>filter_temps),
                                qeos=True)

        # Converting density to ion number density.
        opp_ses_data['idens'] = ((opp.NA * opp_ses_data['ele_dens'])
                                            / opp_ses_data['abar'])


        # Adjust for Znum, Xnum, and Anum.
        if Znum is None:
            if 'Znum' in opp_ses_data:
                Znum = opp_ses_data['Znum']
            else:
                raise ValueError('Znum Varray should be provided!')
        if type(Znum) is int:
            Znum = [Znum]
        opp_ses_data['Znum'] = np.array(Znum, dtype='int')

        if Anum is None:
            opp_ses_data['Anum'] = np.array([ptab.elements[el].mass for el in opp_ses_data['Znum']])
        else:
            opp_ses_data['Anum'] = np.array(Anum)
        opp_ses_data['Zsymb'] = np.array([ptab.elements[el].symbol for el in opp_ses_data['Znum']], dtype='|S2')

        if Xnum is None:
            if len(Znum) == 1:
                opp_ses_data['Xnum'] = np.array([1.0])
            else:
                raise ValueError('Xnum array should be provided!')
        else:
            opp_ses_data['Xnum'] = np.array(Xnum)


        #ionization is possibly on a completely different grid
        if (not 'zbar' in opp_ses_data.keys()):
            # Calculate zbar using thomas_fermi_ionization.
            # If there are multiple elements, it suffices to use the average
            # atomic number in this calculation - JTL
            dens_arr, temp_arr = np.meshgrid(opp_ses_data['ele_dens'],
                                             opp_ses_data['ele_temps'])
            zbar = eos.thomas_fermi_ionization(dens_arr,
                                               temp_arr,
                                               opp_ses_data['Znum'].mean(),
                                               opp_ses_data['abar']).T
            opp_ses_data['zbar'] = zbar

        # Translating SESAME names to common dictionary format.
        if qeos:
            # Names are slightly different for QEOS SESAME
            names_dict = {'idens'     :'idens'  ,
                          'ele_temps' :'temp'   , # We merged ele_ and ion_ dens &
                                                 # temp grids for qeos.
                          'ele_dens'  :'dens'   ,
                          'zbar'      :'Zf_DT'  ,
                          'total_eint':'Ut_DT'  , # But not their energies.
                          'ele_eint'  :'Uec_DT' ,
                          'ion_eint'  :'Ui_DT'  ,
                          'ion_pres'  :'Pi_DT'  ,
                          'ele_pres'  :'Pec_DT' ,
                          'Znum'      :'Znum'   ,
                          'Xnum'      :'Xnum'   ,
                          'bulkmod'   :'BulkMod',
                          'abar'      :'Abar'   ,
                          'zmax'      :'Zmax'
                          }

        else:
            names_dict = {'idens':'idens',
                          'ele_temps':'temp', # We merged ele_ and ioncc_ dens &
                                              # temp grids.
                          'ele_dens':'dens',
                          'zbar':'Zf_DT',
                          'total_eint':'Ut_DT', # But not their energies.
                          'ele_eint':'Uec_DT',
                          'ioncc_eint':'Ui_DT',
                          'ioncc_pres':'Pi_DT',
                          'ele_pres':'Pec_DT',
                          'Znum':'Znum',
                          'Xnum':'Xnum',
                          'bulkmod':'BulkMod',
                          'abar':'Abar',
                          'zmax':'Zmax'
                          }

        # Initialize dictionary.
        eos_dict = {}

        # Creating the tables.
        for ses_key, eos_key in sorted(names_dict.items()):
            try:
                eos_dict[eos_key] = opp_ses_data[ses_key]
            except KeyError:
                print('No data for {} is being written.'.format(eos_key))

        # Handle the logarithmic data.
        if log is not None:
            for key in eos_dict.keys():
                if key in log:
                    eos_dict[key] = np.log10(eos_dict[key])
        return eos_dict
def writeSesameFile(fn, t201, t301, t303, t304, t305, t502, t504, t505, t601):
    CHAR_LINE_LEN = 80
    WORDS_PER_LINE = 5
    comment = 'This Sesame table was generated using OpacPlot2 to convert an IONMIX EoS table to the Sesame format'
    f = open(fn, 'w')
    #write comments
    comment = comment + ' '*(80-len(comment)%80)
    header = ' 0  9999   101   %d   r    82803    22704   1' % len(comment)
    header = header + (79-len(header))*' ' + '0\n'
    f.write(header)
    for i in range(len(comment)//80):
        f.write(comment[i*80:(i+1)*80]+'\n')

    def theader(num):
        return(' 1  9999   %d   240   r    82803    22704   1                                 1\n'  % num)
    def write_block(fid, num, data):
        header = ' 1  9999   %d   %d   r    82803    22704   1'  % (num,len(data))
        header = header + (79-len(header))*' ' + '1\n'
        fid.write(header)
        count = 0
        for n in range(len(data)):
            count += 1
            fid.write('%22.15E' % data[n])
            if count == 5:
                count = 0
                fid.write('11111\n')
        if count > 0:
            m = 5-count
            fid.write(m*22*' ' + count*'1' + m*'0' + '\n')

    write_block(f, 201, t201)
    write_block(f, 301, t301)
    write_block(f, 303, t303)
    write_block(f, 304, t304)
    write_block(f, 305, t305)
    write_block(f, 502, t502)
    write_block(f, 504, t504)
    write_block(f, 505, t505)
    write_block(f, 601, t601)
    f.close()
