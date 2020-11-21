# 2020/Nov/5
# 2020/Oct/11
# No need to add object like class Force(object) in Python 3

import simtk.openmm.app
import simtk.openmm
import simtk.unit as unit
import numpy as np
import itertools

_ef = 1 * unit.kilocalorie / unit.kilojoule  # energy scaling factor
_df = 1 * unit.angstrom / unit.nanometer  # distance scaling factor
_af = 1 * unit.degree / unit.radian  # angle scaling factor
_complement = {'DA': 'DT', 'DT': 'DA', 'DG': 'DC', 'DC': 'DG', 'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
_dnaResidues = ['DA', 'DC', 'DT', 'DG']
_rnaResidues = ['A', 'C', 'U', 'G']
_proteinResidues = ['IPR', 'IGL', 'NGP']

class Force():
    """ Wrapper for the openMM force. """

    def __init__(self, nucleic):
        self.periodic = nucleic.periodic
        self.force = None
        self.nucleic = nucleic

        # Followed by Carlos's design, every forces should have reset and defineInteraction functions.
        # Define the dna force
        self.reset()

        # Define the interaction pairs
        self.defineInteraction()

    #Called when an attribute lookup has not found the attribute in the usual places 
    #(i.e. it is not an instance attribute nor is it found in the class tree for self). 
    # name is the attribute name. This method should return the (computed) attribute value or raise an AttributeError exception.
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif 'force' in self.__dict__:
            try:
                return getattr(self.force, attr)
            except:
                pass
        else:
            if '__repr__' in self.__dict__:
                raise AttributeError(f"type object {str(self)} has no attribute {str(attr)}")
            else:
                raise AttributeError()


class Backbone(Force, simtk.openmm.CustomBondForce):
    def __init__(self, nucleic, force_group=1):
        self.force_group = force_group
        super().__init__(nucleic)

    def getParameterNames(self):
        self.perInteractionParameters = []
        self.GlobalParameters = []
        for i in range(self.force.getNumPerBondParameters()):
            self.perInteractionParameters += [self.force.getPerBondParameterName(i)]
        for i in range(self.force.getNumGlobalParameters()):
            self.GlobalParameters += [self.force.getGlobalParameterName(i)]
        return [self.perInteractionParameters, self.GlobalParameters]

    def reset(self):
        bondForce = simtk.openmm.CustomBondForce("Kb2*(r-r0)^2+Kb3*(r-r0)^3+Kb4*(r-r0)^4")
        bondForce.addPerBondParameter('r0')
        bondForce.addPerBondParameter('Kb2')
        bondForce.addPerBondParameter('Kb3')
        bondForce.addPerBondParameter('Kb4')
        bondForce.setUsesPeriodicBoundaryConditions(self.periodic)
        bondForce.setForceGroup(self.force_group)
        self.force = bondForce

        # epsilon = 2
        # delta = 0.25
        # backbone = 0.76 * 8.4

    def defineInteraction(self):
        for i, b in self.nucleic.bonds.iterrows():
            # Units converted from
            parameters = [b['r0'] * _df,
                          b['Kb2'] / _df ** 2 * _ef,
                          b['Kb3'] / _df ** 3 * _ef,
                          b['Kb4'] / _df ** 4 * _ef]
            #parameters = [2, 0.76*8.4 * _df, 0.25 * _df]
            self.force.addBond(int(b['aai']), int(b['aaj']), parameters)

class Angle(Force, simtk.openmm.HarmonicAngleForce):
    def __init__(self, nucleic, force_group=2):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        angleForce = simtk.openmm.HarmonicAngleForce()
        angleForce.setUsesPeriodicBoundaryConditions(self.periodic)
        angleForce.setForceGroup(self.force_group)
        self.force = angleForce

    def defineInteraction(self):
        for i, a in self.nucleic.angles.iterrows():
            parameters = [a['t0'] * _af,
                          a['epsilon'] * 2]
            self.force.addAngle(int(a['aai']), int(a['aaj']), int(a['aak']), *parameters)



class Dihedral(Force, simtk.openmm.CustomTorsionForce):
    def __init__(self, nucleic, force_group=9):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        dihedralForce = simtk.openmm.CustomTorsionForce("""K_gaussian*exp(-dt_periodic^2/2/sigma^2);
                                                      cs=cos(dt);
                                                      dt_periodic=dt-floor(dt/(2*pi))*(2*pi);
                                                      dt=theta-t0""")
        # Use floor function to deal with the NaN problem in atan function
        dihedralForce.setUsesPeriodicBoundaryConditions(self.periodic)
        dihedralForce.addPerTorsionParameter('K_gaussian')
        dihedralForce.addPerTorsionParameter('sigma')
        dihedralForce.addPerTorsionParameter('t0')
        dihedralForce.addGlobalParameter('pi', np.pi)
        dihedralForce.setForceGroup(self.force_group)
        self.force = dihedralForce

    def defineInteraction(self):
        for i, a in self.nucleic.dihedrals.iterrows():
            parameters = [a['K_gaussian'] * _ef,
                          a['sigma'],
                          (a['t0']) * _af]
            particles = [a['aai'], a['aaj'], a['aak'], a['aal']]
            self.force.addTorsion(*particles, parameters)


class BaseStacking(Force, simtk.openmm.CustomCompoundBondForce):
    def __init__(self, nucleic, force_group=8):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        stackingForce = simtk.openmm.CustomCompoundBondForce(3, """rep+f2*attr;
                                                                rep=epsilon*(1-exp(-alpha*(dr)))^2*step(-dr);
                                                                attr=epsilon*(1-exp(-alpha*(dr)))^2*step(dr)-epsilon;
                                                                dr=distance(p2,p3)-sigma;
                                                                f2=max(f*pair2,pair1);
                                                                pair1=step(dt+pi/2)*step(pi/2-dt);
                                                                pair2=step(dt+pi)*step(pi-dt);
                                                                f=1-cos(dt)^2;
                                                                dt=rng*(angle(p1,p2,p3)-t0);""")
        stackingForce.setUsesPeriodicBoundaryConditions(self.periodic)
        stackingForce.addPerBondParameter('epsilon')
        stackingForce.addPerBondParameter('sigma')
        stackingForce.addPerBondParameter('t0')
        stackingForce.addPerBondParameter('alpha')
        stackingForce.addPerBondParameter('rng')
        stackingForce.addGlobalParameter('pi', np.pi)
        stackingForce.setForceGroup(self.force_group)
        self.force = stackingForce

    def defineInteraction(self):
        for i, a in self.nucleic.stackings.iterrows():
            parameters = [a['epsilon'] * _ef,
                          a['sigma'] * _df,
                          a['t0'] * _af,
                          a['alpha'] / _df,
                          a['rng']]
            self.force.addBond([a['aai'], a['aaj'], a['aak']], parameters)

def addNonBondedExclusions(nucleic, force, OpenCLPatch=True):
    is_rna = nucleic.atoms['resname'].isin(_rnaResidues)
    atoms = nucleic.atoms.copy()
    selection = atoms[is_rna]
    for (i, atom_a), (j, atom_b) in itertools.combinations(selection.iterrows(), r=2):
        if j < i:
            i, j = j, i
            atom_a, atom_b = atom_b, atom_a
        # Neighboring residues
        if atom_a.chainID == atom_b.chainID and (abs(atom_a.resSeq - atom_b.resSeq) <= 1):
            force.addExclusion(i, j)
            # print(i, j)
        # Base-pair residues
        elif OpenCLPatch and (atom_a['name'] in _complement.keys()) and (atom_b['name'] in _complement.keys()) and (
                atom_a['name'] == _complement[atom_b['name']]):
            force.addExclusion(i, j)
            # print(i, j)

class Exclusion(Force, simtk.openmm.CustomNonbondedForce):
    def __init__(self, nucleic, force_group = 12):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        exclusionForce = simtk.openmm.CustomNonbondedForce("""energy;
                                                            energy=(epsilon*((sigma/r)^12-2*(sigma/r)^6)+epsilon)*step(sigma-r);
                                                            sigma=0.5*(sigma1+sigma2); 
                                                            epsilon=sqrt(epsilon1*epsilon2)""")
        exclusionForce.addPerParticleParameter('epsilon')
        exclusionForce.addPerParticleParameter('sigma')
        exclusionForce.setCutoffDistance(1.8)
        exclusionForce.setForceGroup(self.force_group)  # There can not be multiple cutoff distance on the same force group
        if self.periodic:
            exclusionForce.setNonbondedMethod(exclusionForce.CutoffPeriodic)
        else:
            exclusionForce.setNonbondedMethod(exclusionForce.CutoffNonPeriodic)
        self.force = exclusionForce

    def defineInteraction(self):
        # addParticles
        particle_definition = self.nucleic.particle_definition[self.nucleic.particle_definition['nucleic'] == self.nucleic.na_type]
        particle_definition.index = particle_definition.name

        # Reduces or increases the cutoff to the maximum particle radius
        self.force.setCutoffDistance(particle_definition.radius.max() * _df)

        # Select only rna atoms
        is_rna = self.nucleic.atoms['resname'].isin(_rnaResidues)
        atoms = self.nucleic.atoms.copy()
        atoms['is_rna'] = is_rna
        for i, atom in atoms.iterrows():
            if atom.is_rna:
                param = particle_definition.loc[atom['name']]
                parameters = [param.epsilon * _ef,
                              param.radius * _df]
            else:
                parameters = [0, .1]  # Null energy and some radius)
            # print(i, parameters)
            self.force.addParticle(parameters)

        # addExclusions
        addNonBondedExclusions(self.nucleic, self.force)


class Electrostatics(Force, simtk.openmm.CustomNonbondedForce):
    def __init__(self, nucleic, force_group=13, temperature=300*unit.kelvin, salt_concentration=100*unit.millimolar):
        self.force_group = force_group
        self.T = temperature
        self.C = salt_concentration
        super().__init__(nucleic)

    def reset(self):
        T = self.T
        C = self.C
        e = 249.4 - 0.788 * (T / unit.kelvin) + 7.2E-4 * (T / unit.kelvin) ** 2
        a = 1 - 0.2551 * (C / unit.molar) + 5.151E-2 * (C / unit.molar) ** 2 - 6.889E-3 * (C / unit.molar) ** 3
        #print(e, a)
        dielectric = e * a
        # Debye length
        kb = simtk.unit.BOLTZMANN_CONSTANT_kB  # Bolztmann constant
        Na = simtk.unit.AVOGADRO_CONSTANT_NA  # Avogadro number
        ec = 1.60217653E-19 * unit.coulomb  # proton charge
        pv = 8.8541878176E-12 * unit.farad / unit.meter  # dielectric permittivity of vacuum

        ldby = np.sqrt(dielectric * pv * kb * T / (2.0 * Na * ec ** 2 * C))
        ldby = ldby.in_units_of(unit.nanometer)
        denominator = 4 * np.pi * pv * dielectric / (Na * ec ** 2)
        denominator = denominator.in_units_of(unit.kilocalorie_per_mole**-1 * unit.nanometer**-1)
        #print(ldby, denominator)

        electrostaticForce = simtk.openmm.CustomNonbondedForce("""energy;
                                                                energy=q1*q2*exp(-r/dh_length)/denominator/r;""")
        electrostaticForce.addPerParticleParameter('q')
        electrostaticForce.addGlobalParameter('dh_length', ldby)
        electrostaticForce.addGlobalParameter('denominator', denominator)

        electrostaticForce.setCutoffDistance(5)
        if self.periodic:
            electrostaticForce.setNonbondedMethod(electrostaticForce.CutoffPeriodic)
        else:
            electrostaticForce.setNonbondedMethod(electrostaticForce.CutoffNonPeriodic)
        electrostaticForce.setForceGroup(self.force_group)
        self.force = electrostaticForce

    def defineInteraction(self):
        # addParticles
        particle_definition = self.nucleic.particle_definition[self.nucleic.particle_definition['nucleic'] == self.nucleic.na_type]
        particle_definition.index = particle_definition.name

        # Select only rna atoms
        is_rna = self.nucleic.atoms['resname'].isin(_rnaResidues)
        atoms = self.nucleic.atoms.copy()
        atoms['is_rna'] = is_rna

        for i, atom in atoms.iterrows():
            if atom.is_rna:
                param = particle_definition.loc[atom['name']]
                parameters = [param.charge]
            else:
                parameters = [0]  # No charge if it is not DNA
            # print (i,parameters)
            self.force.addParticle(parameters)

        # add neighbor exclusion
        addNonBondedExclusions(self.nucleic, self.force, self.OpenCLPatch)


class BaseStacking(Force, simtk.openmm.CustomCompoundBondForce):
    def __init__(self, nucleic, force_group=9):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        stackingForce = simtk.openmm.CustomCompoundBondForce(3, """rep+f2*attr;
                                                                rep=epsilon*(1-exp(-alpha*(dr)))^2*step(-dr);
                                                                attr=epsilon*(1-exp(-alpha*(dr)))^2*step(dr)-epsilon;
                                                                dr=distance(p2,p3)-sigma;
                                                                f2=max(f*pair2,pair1);
                                                                pair1=step(dt+pi/2)*step(pi/2-dt);
                                                                pair2=step(dt+pi)*step(pi-dt);
                                                                f=1-cos(dt)^2;
                                                                dt=rng*(angle(p1,p2,p3)-t0);""")
        stackingForce.setUsesPeriodicBoundaryConditions(self.periodic)
        stackingForce.addPerBondParameter('epsilon')
        stackingForce.addPerBondParameter('sigma')
        stackingForce.addPerBondParameter('t0')
        stackingForce.addPerBondParameter('alpha')
        stackingForce.addPerBondParameter('rng')
        stackingForce.addGlobalParameter('pi', np.pi)
        stackingForce.setForceGroup(self.force_group)
        self.force = stackingForce

    def defineInteraction(self):
        for i, a in self.nucleic.stackings.iterrows():
            parameters = [a['epsilon'] * _ef,
                          a['sigma'] * _df,
                          a['t0'] * _af,
                          a['alpha'] / _df,
                          a['rng']]
            self.force.addBond([a['aai'], a['aaj'], a['aak']], parameters)


import numpy as np
def readDotBacket(secondary_file="dot_bracket.txt", output_file="contact_list.txt"):
    with open(secondary_file, 'r') as fopen:
        lines = fopen.readlines()
        sequence = lines[0]
    
    parentheses = []
    brackets = []

    index = 1
    with open(output_file, 'w') as fwrite:
        for i in sequence3:
            if i == '(':
                parentheses.append(index)
            elif i == '[':
                brackets.append(index)
            elif i == ')':
                #print("%s %s" %(parentheses.pop(), index))
                fwrite.write("%s %s\n" %(parentheses.pop(), index))
            elif i == ']':
                #print("%s %s" %(brackets.pop(), index))
                fwrite.write("%s %s\n" %(brackets.pop(), index))
            index += 1
    return output_file

class BasePair(Force, simtk.openmm.CustomHbondForce):
    def __init__(self, nucleic, force_group=10, OpenCLPatch=True, contact_list='contact_list.txt'):
        self.force_group = force_group
        self.contact_list = contact_list
        self.OpenCLPatch = OpenCLPatch
        super().__init__(nucleic)

    def reset(self):
        def basePairForce():
            pairForce = simtk.openmm.CustomHbondForce('''temp;temp=rep+1/2*(1+cos(dphi))*fdt1*fdt2*attr;
                                                         rep  = epsilon*(1-exp(-alpha*dr))^2*(1-step(dr));
                                                         attr = epsilon*(1-exp(-alpha*dr))^2*step(dr)-epsilon;
                                                         fdt1 = max(f1*pair0t1,pair1t1);
                                                         fdt2 = max(f2*pair0t2,pair1t2);
                                                         pair1t1 = step(pi/2+dt1)*step(pi/2-dt1);
                                                         pair1t2 = step(pi/2+dt2)*step(pi/2-dt2);
                                                         pair0t1 = step(pi+dt1)*step(pi-dt1);
                                                         pair0t2 = step(pi+dt2)*step(pi-dt2);
                                                         f1 = 1-cos(dt1)^2;
                                                         f2 = 1-cos(dt2)^2;
                                                         dphi = dihedral(d2,d1,a1,a2)-phi0;
                                                         dr    = distance(d1,a1)-sigma;
                                                         dt1   = rng*(angle(d2,d1,a1)-t01);
                                                         dt2   = rng*(angle(a2,a1,d1)-t02);''')
            if self.periodic:
                pairForce.setNonbondedMethod(pairForce.CutoffPeriodic)
            else:
                pairForce.setNonbondedMethod(pairForce.CutoffNonPeriodic)
            pairForce.setCutoffDistance(1.8)  # Paper
            pairForce.addPerDonorParameter('phi0')
            pairForce.addPerDonorParameter('sigma')
            pairForce.addPerDonorParameter('t01')
            pairForce.addPerDonorParameter('t02')
            pairForce.addPerDonorParameter('rng')
            pairForce.addPerDonorParameter('epsilon')
            pairForce.addPerDonorParameter('alpha')
            pairForce.addGlobalParameter('pi', np.pi)
            self.force = pairForce
            pairForce.setForceGroup(self.force_group)
            return pairForce

        basePairForces = {}
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['nucleic'] == self.nucleic.na_type]
        for i, pair in pair_definition.iterrows():
            basePairForces.update({i: basePairForce()})
        self.forces = basePairForces

    def defineInteraction(self):
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['nucleic'] == self.nucleic.na_type]
        atoms = self.nucleic.atoms.copy()
        atoms['index'] = atoms.index
        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'])
        is_rna = atoms['resname'].isin(_rnaResidues)


        contact_list = np.loadtxt(self.contact_list)
        
        for each_contact_pair in contact_list:
            D1 = atoms[(atoms['resSeq'] == int(each_contact_pair[0])) & is_rna].copy()
            D1 = D1[D1['name'].isin(['A','U','C','G'])]
            A1 = atoms[(atoms['resSeq'] == int(each_contact_pair[1])) & is_rna].copy()
            A1 = A1[A1['name'].isin(['A','U','C','G'])]

            try:
                D2 = atoms.loc[[(c, r, 'S') for c, r, n in D1.index]]
            except KeyError:
                for c, r, n in D1.index:
                    if (c, r, 'S') not in atoms.index:
                        print(f'Residue {c}:{r} does not have a Sugar atom (S)')
                raise KeyError

            try:
                A2 = atoms.loc[[(c, r, 'S') for c, r, n in A1.index]]
            except KeyError:
                for c, r, n in A1.index:
                    if (c, r, 'S') not in atoms.index:
                        print(f'Residue {c}:{r} does not have a Sugar atom (S)')
                raise KeyError
            
            print("%s %s"%(str(D1['name'][0]), str(A1['name'][0])))
            correct_pair = False
            for i, pair in pair_definition.iterrows():
                if (pair['Base1'] == D1['name'][0]) & (pair['Base2'] == A1['name'][0]):
                    correct_pair = True
                    break
            if correct_pair == False:
                break

            #print(pair.Base1)
            #print(i)
            D1_list = list(D1['index'])
            print(D1_list)
            print(pair)
            A1_list = list(A1['index'])
            D2_list = list(D2['index'])
            A2_list = list(A2['index'])

            # Define parameters
            parameters = [pair.torsion * _af,
                        pair.sigma * _df,
                        pair.t1 * _af,
                        pair.t2 * _af,
                        pair.rang,
                        pair.epsilon * _ef,
                        pair.alpha / _df]

            # Add donors and acceptors
            # Here I am including the same atom twice,
            # it doesn't seem to break things
            for d1, d2 in zip(D1_list, D2_list):
                self.forces[i].addDonor(d1, d2, d2, parameters)
                #print(d1, d2, d2, parameters)
            for a1, a2 in zip(A1_list, A2_list):
                self.forces[i].addAcceptor(a1, a2, a2)
                #print(a1, a2, a2)
            # Exclude interactions
            D1['donor_id'] = [i for i in range(len(D1))]
            A1['aceptor_id'] = [i for i in range(len(A1))]

            for (_i, atom_a), (_j, atom_b) in itertools.product(D1.iterrows(), A1.iterrows()):
                # Neighboring residues
                # The sequence exclusion was reduced to two residues
                # since the maximum number of exclusions in OpenCL is 4.
                # In the original 3SPN2 it was 3 residues (6 to 9)
                # This change has no noticeable effect
                if (atom_a.chainID == atom_b.chainID) and (abs(atom_a.resSeq - atom_b.resSeq) <= 2):
                    self.forces[i].addExclusion(atom_a['donor_id'], atom_b['aceptor_id'])
                    #print(_i, _j)

    def addForce(self, system):
        for f in self.forces:
            system.addForce(self.forces[f])
