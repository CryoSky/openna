import sys
import pandas

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

def writePDB(data, pdb_file='clean.pdb'):
    """ Writes a minimal version of the pdb file needed for openmm """
    # Compute chain field
    if type(data['chainID'].iloc[0]) is not str:
        chain_ix = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        data['chainID'] = [chain_ix[i - 1] for i in data['chainID']]
        # Write pdb file
    with open(pdb_file, 'w+') as pdb:
        for i, atom in data.iterrows():
            pdb_line = f'ATOM  {i + 1:>5} {atom["name"]:^4} {atom.resname:<3} {atom.chainID}{atom.resSeq:>4}    {atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}{atom.occupancy:>6.2f}{atom.tempFactor:>6.2f}' + ' ' * 10 + f'{atom.element:>2}' + ' ' * 2 # modified to meet RNA requirement
            assert len(pdb_line) == 80, 'An item in the atom table is longer than expected'
            pdb.write(pdb_line + '\n')
    return pdb_file


input = sys.argv[1]
output = sys.argv[2]

test1 = parsePDB(input)
test2 = test1[test1['resname'].isin(['A', 'U', 'C', 'G'])]
test4 = test2.reset_index()
test4['serial'] = test4.index+1
test5 = test4.drop(columns=['index'])
test6 = test5[test5['recname'] != 'HETATM']
output = writePDB(test6, pdb_file='clean.pdb')
