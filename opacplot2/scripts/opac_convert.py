import opacplot2 as opp
import argparse
import os.path
import numpy as np
from opacplot2.constants import EV_TO_KELVIN, ERGCC_TO_GPA, ERGG_TO_MJKG


def get_input_data():
    # Available formats.
    avail_output_formats = ["ionmix", "sesame"]
    avail_input_formats = ["propaceos", "multi", "sesame", "sesame-qeos", "ionmix"]

    # Creating the argument parser.
    parser = argparse.ArgumentParser(
        description="This script is used to browse various"
        "EoS/Opacity tables formats."
    )

    parser.add_argument(
        "-v", "--verbose", action="store_const", const=True, help="Verbosity option."
    )

    parser.add_argument(
        "--Znum",
        action="store",
        type=str,
        help="Comma separated list of Z" "numbers for every component.",
    )

    parser.add_argument("--mpi", action="store", type=str, help="Mass per ion in grams")

    parser.add_argument(
        "--Xfracs",
        action="store",
        type=str,
        help="Comma separated list of X" "fractions for every component.",
    )

    parser.add_argument(
        "--outname",
        action="store",
        type=str,
        help="Name for output file without extension.",
    )

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        choices=avail_input_formats,
        help="Input filetype.",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        choices=avail_output_formats,
        default="ionmix",
        help="Output filetype. Default: IONMIX.",
    )

    parser.add_argument("input_file", action="store", type=str, help="Input file.")

    parser.add_argument(
        "--log", action="store", type=str, help="Logarithmic data keys."
    )

    parser.add_argument(
        "--tabnum", action="store", type=str, help="Specify the SESAME table number."
    )

    args = parser.parse_args()

    # Get the relevant paths and filenames.
    path_in = os.path.abspath(args.input_file)
    basedir, fn_in = os.path.split(path_in)
    # Split filename twice in case of MULTI files (.opr.gz, etc)
    basename = os.path.splitext(os.path.splitext(fn_in)[0])[0]

    # Adjusting the input.
    # Set the base output name that specified by --outname.
    # Otherwise, set it to the same as the base input name.
    if args.outname is not None:
        args.outname = os.path.splitext(os.path.abspath(args.outname))[0]
    else:
        args.outname = os.path.join(basedir, basename)

    # Create lists out of the strings for Znum, Xfracs, mpi, and log if given.
    if args.Znum is not None:
        args.Znum = [int(num) for num in args.Znum.split(",")]
    if args.Xfracs is not None:
        args.Xfracs = [float(num) for num in args.Xfracs.split(",")]
    if args.mpi is not None:
        args.mpi = [float(num) for num in args.mpi.split(",")]
    if args.log is not None:
        args.log = [str(key) for key in args.log.split(",")]

    # Convert tabnum into int.
    if args.tabnum is not None:
        try:
            args.tabnum = int(args.tabnum)
        except ValueError:
            raise ValueError("Please provide a valid SESAME table number.")

    input_data = {
        "args": args,
        "basename": basename,
        "path_in": path_in,
        "basedir": basedir,
        "fn_in": fn_in,
    }

    return input_data


def read_format_ext(args, fn_in):
    # Try to read from the input file extension.
    ext_dict = {
        ".prp": "propaceos",
        ".eps": "multi",
        ".opp": "multi",
        ".opz": "multi",
        ".opr": "multi",
        ".cn4": "ionmix",
        ".mexport": "sesame-qeos",
        ".ses": "sesame",
        ".html": "tops",
        ".tops": "tops",
    }
    # If the input file is compressed, choose the next extension.
    if os.path.splitext(fn_in)[1] == ".gz":
        _, ext = os.path.splitext(os.path.splitext(fn_in)[0])
    else:
        _, ext = os.path.splitext(fn_in)

    # Choose the correct input type based on extension and set args.input
    # accordingly.
    if ext in ext_dict.keys():
        args.input = ext_dict[ext]
    else:
        raise Warning(
            "Cannot tell filetype from extension. Please specify "
            "input file type with --input."
        )


class Formats_toEosDict(object):
    """
    Contains handling functions to convert different types of tables
    into a common EoS dictionary for IONMIX.
    """

    def __init__(self, args, basedir, basename, path_in):
        # Initialize the dictionary for handling functions.
        self.set_handle_dict()

        # Set attributes.
        self.args = args
        self.basedir = basedir
        self.basename = basename
        self.path_in = path_in

        # Use handle_dict to create the eos_dict based on the input format.
        try:
            self.eos_dict = self.handle_dict[args.input]()
        except KeyError:
            raise KeyError("Must use valid format name.")

    def set_handle_dict(self):
        self.handle_dict = {
            "propaceos": self.propaceos_toEosDict,
            "multi": self.multi_toEosDict,
            "sesame": self.sesame_toEosDict,
            "ionmix": self.ionmix_toEosDict,
            "tops": self.tops_toEosDict,
            "sesame-qeos": self.sesame_qeos_toEosDict,
        }

    def propaceos_toEosDict(self):
        # If we are unable to find the correct library for opg_propaceos
        # we need to let the user know.
        try:
            import opacplot2.opg_propaceos

            op = opp.opg_propaceos.OpgPropaceosAscii(self.path_in, mpi=self.args.mpi)
            eos_dict = op.toEosDict(log=self.args.log)
            return eos_dict
        except ImportError:
            raise ImportError("You do not have the opg_propaceos script.")

    def multi_toEosDict(self):
        op = opp.OpgMulti.open_file(self.basedir, self.basename)
        eos_dict = op.toEosDict(
            Znum=self.args.Znum, Xnum=self.args.Xfracs, log=self.args.log
        )
        return eos_dict

    def ionmix_toEosDict(self):
        op = opp.OpacIonmix(self.path_in, self.args.mpi, man=True, twot=True)
        eos_dict = op.toEosDict(
            Znum=self.args.Znum, Xnum=self.args.Xfracs, log=self.args.log
        )

        return eos_dict

    def sesame_toEosDict(self):
        try:
            op = opp.OpgSesame(self.path_in, opp.OpgSesame.SINGLE)
        except ValueError:
            op = opp.OpgSesame(self.path_in, opp.OpgSesame.DOUBLE)

        if len(op.data.keys()) > 1:
            raise Warning(
                "More than one material ID found. "
                "Use sesame-extract to create a file "
                "with only one material first."
            )

        if self.args.tabnum is not None:
            eos_dict = op.toEosDict(
                Znum=self.args.Znum,
                Xnum=self.args.Xfracs,
                log=self.args.log,
                tabnum=self.args.tabnum,
            )
        else:
            eos_dict = op.toEosDict(
                Znum=self.args.Znum, Xnum=self.args.Xfracs, log=self.args.log
            )
        return eos_dict

    def sesame_qeos_toEosDict(self):
        raise Warning("QEOS-SESAME is not ready yet!")
        try:
            op = opp.OpgSesame(self.path_in, opp.OpgSesame.SINGLE)
        except ValueError:
            op = opp.OpgSesame(self.path_in, opp.OpgSesame.DOUBLE)

        if len(op.data.keys()) > 1:
            raise Warning(
                "More than one material ID found. "
                "Use sesame-extract to create a file "
                "with only one material first."
            )

        if self.args.tabnum is not None:
            eos_dict = op.toEosDict(
                Znum=self.args.Znum,
                Xnum=self.args.Xfracs,
                qeos=True,
                log=self.args.log,
                tabnum=self.args.tabnum,
            )
        else:
            eos_dict = op.toEosDict(
                Znum=self.args.Znum, Xnum=self.args.Xfracs, qeos=True, log=self.args.log
            )
        return eos_dict

    def tops_toEosDict(self):
        op = opp.OpgTOPS(self.path_in)
        eos_dict = op.toEosDict(fill_eos=True)
        return eos_dict


class EosDict_toSesameFile(object):
    """
    Takes a common EoS dictionary and writes it to the correct output format.
    """

    def __init__(self, args, eos_dict):
        self.set_handle_dict()
        self.args = args
        self.eos_dict = eos_dict

        self.handle_dict[args.output]()

    def set_handle_dict(self):
        self.handle_dict = {"sesame": self.eosDict_toSesame}

    def eosDict_toSesame(self):
        # initialize sesame argument dictionary
        ses_dict = {}
        # we should need to convert units to what sesame needs
        dens = np.array(self.eos_dict["dens"])
        temp = np.array(self.eos_dict["temp"])
        pele = np.array(self.eos_dict["Pec_DT"])
        pion = np.array(self.eos_dict["Pi_DT"])
        uele = np.array(self.eos_dict["Uec_DT"])
        uion = np.array(self.eos_dict["Ui_DT"])
        utot = np.array(self.eos_dict["Ut_DT"])
        dummy = np.array(self.eos_dict["Ut_DT"])
        zbar = np.array(self.eos_dict["Zf_DT"])
        plnk = np.array(self.eos_dict["opp_int"])
        rsln = np.array(self.eos_dict["opr_int"])
        ptot = pele + pion

        if len(self.eos_dict["Znum"]) > 1:
            zz = self.eos_dict["Znum"]
            xx = self.eos_dict["Xnum"]
            znum = 0.0
            for i in range(len(self.eos_dict["Znum"])):
                znum += zz[i] * xx[i]
        else:
            znum = self.eos_dict["Znum"][0]

        ses_dict["t201"] = np.array([znum, self.eos_dict["Abar"], 1.0, 1.0, 1.0])
        ses_dict["t301"] = self.tables_toSesame(dens, temp, ptot, utot, dummy)
        ses_dict["t303"] = self.tables_toSesame(dens, temp, pion, uion, dummy)
        ses_dict["t304"] = self.tables_toSesame(dens, temp, pele, uele, dummy)
        ses_dict["t305"] = self.tables_toSesame(dens, temp, pion, uion, dummy)
        ses_dict["t502"] = self.zbar_toSesame(dens, temp, rsln)
        ses_dict["t505"] = self.zbar_toSesame(dens, temp, plnk)
        ses_dict["t504"] = self.zbar_toSesame(dens, temp, zbar)
        ses_dict["t601"] = self.zbar_toSesame(dens, temp, zbar)

        opp.writeSesameFile(self.args.outname + ".ses", **ses_dict)

    def tables_toSesame(self, dens, temp, pres, enrg, fnrg):
        # flatten (n,t) tables into sesame array for 301-305 tables
        ses_tab = np.array([len(dens), len(temp)])
        ses_tab = np.append(ses_tab, dens)
        ses_tab = np.append(ses_tab, temp * EV_TO_KELVIN)
        ses_tab = np.append(ses_tab, np.transpose(pres).flatten() * ERGCC_TO_GPA)
        ses_tab = np.append(ses_tab, np.transpose(enrg).flatten() * ERGG_TO_MJKG)
        ses_tab = np.append(ses_tab, np.transpose(fnrg).flatten())
        return ses_tab

    def zbar_toSesame(self, dens, temp, data):
        Ldens = np.log10(dens)
        Ltemp = np.log10(temp)
        Ldata = np.log10(np.transpose(data).flatten())
        ses_tab = np.array([len(Ldens), len(Ltemp)])
        ses_tab = np.append(ses_tab, Ldens)
        ses_tab = np.append(ses_tab, Ltemp)
        ses_tab = np.append(ses_tab, Ldata)
        return ses_tab


class EosDict_toIonmixFile(object):
    """
    Takes a common EoS dictionary and writes it to the correct output format.
    """

    def __init__(self, args, eos_dict):
        # Initialize the handling function dictionary.
        self.set_handle_dict()

        # Set attributes.
        self.args = args
        self.eos_dict = eos_dict

        # Execute the write function based on output format.
        self.handle_dict[args.output]()

    def set_handle_dict(self):
        self.handle_dict = {"ionmix": self.eosDict_toIonmix}

    def eosDict_toIonmix(self):
        # These are the naming conventions translated to ionmix arguments.
        imx_conv = {
            "Znum": "zvals",
            "Xnum": "fracs",
            "idens": "numDens",
            "temp": "temps",
            "Zf_DT": "zbar",
            "Pi_DT": "pion",
            "Pec_DT": "pele",
            "Ui_DT": "eion",
            "Uec_DT": "eele",
            "groups": "opac_bounds",
            "opr_mg": "rosseland",
            "opp_mg": "planck_absorb",
            "emp_mg": "planck_emiss",
        }

        # Initialize ionmix argument dictionary.
        imx_dict = {}

        # Translating the keys over.
        for key in imx_conv.keys():
            if key in self.eos_dict.keys():
                imx_dict[imx_conv[key]] = self.eos_dict[key]

        # Set ngroups if opacity bounds are present.
        if "opac_bounds" in imx_dict:
            imx_dict["ngroups"] = len(self.eos_dict["groups"]) - 1

        # Check if required FLASH EoS tables are present.
        imx_req_keys = ["zbar", "eion", "eele", "pion", "pele"]
        if not all(key in imx_dict for key in imx_req_keys):
            print(
                "The required EoS data for FLASH is not present...\n"
                "Aborting the IONMIX file creation..."
            )
            raise Warning("Missing EoS data for IONMIX file to run in FLASH")

        # For verbose flag.
        if self.args.verbose:
            verb_conv = {
                "zvals": "Atomic numbers",
                "fracs": "Element fractions",
                "numDens": "Ion number densities",
                "temps": "Temperatures",
                "zbar": "Average ionizations",
                "pion": "Ion pressure",
                "pele": "Electron pressure",
                "eion": "Ion internal energy",
                "eele": "Electron internal energy",
                "opac_bounds": "Opacity bounds",
                "rosseland": "Rosseland mean opacity",
                "planck_absorb": "Absorption Planck mean opacity",
                "planck_emiss": "Emission Planck mean opacity",
                "ngroups": "Number of opacity groups",
            }

            verb_str = "Wrote the following data to IONMIX file:\n"
            i = 0
            for key in imx_dict.keys():
                i = i + 1
                if i == len(imx_dict.keys()):
                    verb_str = verb_str + "{}. {}".format(i, verb_conv[key])
                else:
                    verb_str = verb_str + "{}. {} \n".format(i, verb_conv[key])
            print(verb_str)

        # Write the ionmix file based on what data is stored in imx_dict.
        opp.writeIonmixFile(self.args.outname + ".cn4", **imx_dict)


def convert_tables():
    # Grab the input data.
    input_data = get_input_data()

    # Read the file extension if the user did not specify an input.
    if input_data["args"].input is None:
        read_format_ext(input_data["args"], input_data["fn_in"])

    # Reading in data and converting it to the common dictionary format.
    eos_dict = Formats_toEosDict(
        input_data["args"],
        input_data["basedir"],
        input_data["basename"],
        input_data["path_in"],
    ).eos_dict

    output_type = input_data["args"].output
    if output_type == "sesame":
        EosDict_toSesameFile(input_data["args"], eos_dict)
    else:
        EosDict_toIonmixFile(input_data["args"], eos_dict)


if __name__ == "__main__":
    convert_tables()
