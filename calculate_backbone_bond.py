import sys
import pandas
import numpy as np

_rnaResidues = ['A', 'C', 'U', 'G']

def parsePDB(pdb_file):
    """Transforms the pdb file into a pandas table for easy access and data editing."""

    def pdb_line(line):
        return dict(recname=str(line[0:6]).strip(),  # record name
                    serial=int(line[6:11]),          # atom serial number
                    name=str(line[12:16]).strip(),   # atom name
                    altLoc=str(line[16:17]),         # alternate location indicator
                    resname=str(line[17:20]).strip(),
                    chainID=str(line[21:22]),
                    resSeq=int(line[22:26]),         # residue sequence number
                    iCode=str(line[26:27]),          # code for insertion of residues
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=1.0 if line[54:60].strip() == '' else float(line[54:60]), # set to 1.0 because Plumed RMSD need 1.0
                    tempFactor=1.0 if line[60:66].strip() == '' else float(line[60:66]),
                    element=str(line[76:78]),        # element symbol, right-justified
                    charge=str(line[78:80]))         # charge on the atom, right-justified

    with open(pdb_file, 'r') as pdb:
        lines = []
        for line in pdb:
            if len(line) > 6 and line[:6] in ['ATOM  ', 'HETATM']:
                lines += [pdb_line(line)]
    pdb_atoms = pandas.DataFrame(lines)
    pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                           'resname', 'chainID', 'resSeq', 'iCode',
                           'x', 'y', 'z', 'occupancy', 'tempFactor',
                           'element', 'charge']]
    return pdb_atoms


def CoarseGrain(pdb_table):
    """ Selects RNA atoms from a pdb table and returns a table containing only the coarse-grained atoms for 3SPN2"""
    masses = {"H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994, "P": 30.973762, }
    CG = {"O5\'": 'P', "C5\'": 'S', "C4\'": 'S', "O4\'": 'S', "C3\'": 'S', "O3\'": 'P', "O2\'": 'S',
            "C2\'": 'S', "C1\'": 'S', "O5*": 'P', "C5*": 'S', "C4*": 'S', "O4*": 'S',
            "C3*": 'S', "O3*": 'P', "C2*": 'S', "C1*": 'S', "N1": 'B', "C2": 'B', "O2": 'B',
            "N2": 'B', "N3": 'B', "C4": 'B', "N4": 'B', "C5": 'B', "C6": 'B', "N9": 'B',
            "C8": 'B', "O6": 'B', "N7": 'B', "N6": 'B', "O4": 'B', "C7": 'B', "P": 'P',
            "OP1": 'P', "OP2": 'P', "O1P": 'P', "O2P": 'P', "OP3": 'P', "HO5'": 'P',
            "H5'": 'S', "H5''": 'S', "H4'": 'S', "H3'": 'S', "H2'": 'S', "H2''": 'S',
            "H1'": 'S', "H8": 'B', "H61": 'B', "H62": 'B', 'H2': 'B', 'H1': 'B', 'H21': 'B',
            'H22': 'B', 'H3': 'B', 'H71': 'B', 'H72': 'B', 'H73': 'B', 'H6': 'B', 'H41': 'B',
            'H42': 'B', 'H5': 'B', "HO3'": 'P'}
    cols = ['recname', 'serial', 'name', 'altLoc',
            'resname', 'chainID', 'resSeq', 'iCode',
            'x', 'y', 'z', 'occupancy', 'tempFactor',
            'element', 'charge', 'type']
    temp = pdb_table.copy()

    # Select RNA residues
    temp = temp[temp['resname'].isin(['A', 'U', 'G', 'C'])]

    # Group the atoms by sugar, phosphate or base
    temp['group'] = temp['name'].replace(CG) # Replace the true atom to group name and add in a new column group
    temp = temp[temp['group'].isin(['P', 'S', 'B'])]

    # Move the O3' to the next residue
    for c in temp['chainID'].unique():
        sel = temp.loc[(temp['name'] == "O3\'") & (temp['chainID'] == c), "resSeq"]
        temp.loc[(temp['name'] == "O3\'") & (temp['chainID'] == c), "resSeq"] = list(sel)[1:] + [-1] # add the resseq 1, the last is -1
        sel = temp.loc[(temp['name'] == "O3\'") & (temp['chainID'] == c), "resname"]
        temp.loc[(temp['name'] == "O3\'") & (temp['chainID'] == c), "resname"] = list(sel)[1:] + ["remove"]
    # temp = temp[temp['resSeq'] > 0]
    temp = temp[temp['resname'] != 'remove']

    # Calculate center of mass
    temp['element'] = temp['element'].str.strip()
    temp['mass'] = temp.element.replace(masses).astype(float)
    temp[['x', 'y', 'z']] = (temp[['x', 'y', 'z']].T * temp['mass']).T[['x', 'y', 'z']]
    temp = temp[temp['element'] != 'H']  # Exclude hydrogens
    Coarse = temp.groupby(['chainID', 'resSeq', 'resname', 'group']).sum().reset_index()
    Coarse[['x', 'y', 'z']] = (Coarse[['x', 'y', 'z']].T / Coarse['mass']).T[['x', 'y', 'z']]

    # Set pdb columns
    Coarse['recname'] = 'ATOM'
    Coarse['name'] = Coarse['group']
    Coarse['altLoc'] = ''
    Coarse['iCode'] = ''
    Coarse['charge'] = ''
    # Change name of base to real base
    mask = (Coarse.name == 'B')
    Coarse.loc[mask, 'name'] = Coarse[mask].resname.str[-1]  # takes last letter from the residue name
    Coarse['type'] = Coarse['name']
    # Set element (depends on base)
    Coarse['element'] = Coarse['name'].replace({'P': 'P', 'S': 'H', 'A': 'N', 'U': 'S', 'G': 'C', 'C': 'O'})
    # Remove P from the beggining
    drop_list = []
    for chain in Coarse.chainID.unique():
        sel = Coarse[Coarse.chainID == chain]
        drop_list += list(sel[(sel.resSeq == sel.resSeq.min()) & sel['name'].isin(['P'])].index)
    Coarse = Coarse.drop(drop_list)
    # Renumber
    Coarse.index = range(len(Coarse))
    Coarse['serial'] = Coarse.index
    return Coarse[cols]

input = sys.argv[1]
test1 = parsePDB(input)
test2 = CoarseGrain(test1)

p_atom = test2[test2['name']=='P']
s_atom = test2[test2['name']=='S']
a_atom = test2[test2['name']=='A']
c_atom = test2[test2['name']=='C']
g_atom = test2[test2['name']=='G']
u_atom = test2[test2['name']=='U']
a = list(c_atom.iterrows())

import numpy as np
distance_all = []

for each_item in a:
    index = each_item[1]['resSeq']
    chain_id = each_item[1]['chainID']
    x1 = each_item[1].x
    y1 = each_item[1].y
    z1 = each_item[1].z

    residue2 = s_atom[(s_atom['resSeq'] == index) & (s_atom['chainID'] == chain_id)].iloc[0] # Default, only one residue will be count
    #print(type(residue2))
    x2 = residue2.x
    y2 = residue2.y
    z2 = residue2.z

    xyz_1 = np.array((x1, y1, z1))
    xyz_2 = np.array((x2, y2, z2))

    dist = np.linalg.norm(xyz_2-xyz_1)
    print(dist)
    distance_all.append(dist)