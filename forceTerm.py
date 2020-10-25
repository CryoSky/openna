# 2020/Oct/11
# No need to add object like class Force(object) in Python 3

import simtk.openmm.app
import simtk.openmm
import simtk.unit as unit
import numpy as np

_ef = 1 * unit.kilocalorie / unit.kilojoule  # energy scaling factor
_df = 1 * unit.angstrom / unit.nanometer  # distance scaling factor
_af = 1 * unit.degree / unit.radian  # angle scaling factor

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
            return getattr(self.force, attr)
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
        # dihedralForce = simtk.openmm.CustomTorsionForce("""K_periodic*(1-cs)-K_gaussian*exp(-atan(tan(dt))^2/2/sigma^2);
        #                                               cs=cos(dt);
        #                                               dt=theta-t0""")
        dihedralForce = simtk.openmm.CustomTorsionForce("""K_periodic*(1-cs)-K_gaussian*exp(-dt_periodic^2/2/sigma^2);
                                                      cs=cos(dt);
                                                      dt_periodic=dt-floor((dt+pi)/(2*pi))*(2*pi);
                                                      dt=theta-t0""")
        # dihedralForce=simtk.openmm.CustomTorsionForce("theta/60.")
        dihedralForce.setUsesPeriodicBoundaryConditions(self.periodic)
        dihedralForce.addPerTorsionParameter('K_periodic')
        dihedralForce.addPerTorsionParameter('K_gaussian')
        dihedralForce.addPerTorsionParameter('sigma')
        dihedralForce.addPerTorsionParameter('t0')
        dihedralForce.addGlobalParameter('pi', np.pi)
        dihedralForce.setForceGroup(self.force_group)
        self.force = dihedralForce

    def defineInteraction(self):
        for i, a in self.nucleic.dihedrals.iterrows():
            parameters = [a['K_dihedral'] * _ef,
                          a['K_gaussian'] * _ef,
                          a['sigma'],
                          (180 + a['t0']) * _af]
            particles = [a['aai'], a['aaj'], a['aak'], a['aal']]
            self.force.addTorsion(*particles, parameters)

class Dihedral(Force, simtk.openmm.CustomTorsionForce):
    def __init__(self, nucleic, force_group=9):
        self.force_group = force_group
        super().__init__(nucleic)

    def reset(self):
        dihedralForce = simtk.openmm.CustomTorsionForce("""K_periodic*(1-cs)-K_gaussian*exp(-dt_periodic^2/2/sigma^2);
                                                      cs=cos(dt);
                                                      dt_periodic=dt-floor((dt+pi)/(2*pi))*(2*pi);
                                                      dt=theta-t0""")
        # Use floor function to deal with the NaN problem in atan function
        dihedralForce.setUsesPeriodicBoundaryConditions(self.periodic)
        dihedralForce.addPerTorsionParameter('K_periodic')
        dihedralForce.addPerTorsionParameter('K_gaussian')
        dihedralForce.addPerTorsionParameter('sigma')
        dihedralForce.addPerTorsionParameter('t0')
        dihedralForce.addGlobalParameter('pi', np.pi)
        dihedralForce.setForceGroup(self.force_group)
        self.force = dihedralForce

    def defineInteraction(self):
        for i, a in self.nucleic.dihedrals.iterrows():
            parameters = [a['K_dihedral'] * _ef,
                          a['K_gaussian'] * _ef,
                          a['sigma'],
                          (180 + a['t0']) * _af]
            particles = [a['aai'], a['aaj'], a['aak'], a['aal']]
            self.force.addTorsion(*particles, parameters)


class Stacking(Force, simtk.openmm.CustomCompoundBondForce):
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


class BasePair(Force, simtk.openmm.CustomHbondForce):
    def __init__(self, nucleic, force_group=10, OpenCLPatch=True):
        self.force_group = force_group
        super().__init__(nucleic, OpenCLPatch)

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
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['RNA'] == self.nucleic.na_type]
        for i, pair in pair_definition.iterrows():
            basePairForces.update({i: basePairForce()})
        self.forces = basePairForces

    def defineInteraction(self):
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['RNA'] == self.nucleic.na_type]
        atoms = self.nucleic.atoms.copy()
        atoms['index'] = atoms.index
        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'])
        is_rna = atoms['resname'].isin(_rnaResidues)

        for i, pair in pair_definition.iterrows():
            D1 = atoms[(atoms['name'] == pair['Base1']) & is_rna].copy()
            A1 = atoms[(atoms['name'] == pair['Base2']) & is_rna].copy()

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

            D1_list = list(D1['index'])
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