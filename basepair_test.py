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
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['nucleic'] == self.nucleic.na_type]
        for i, pair in pair_definition.iterrows():
            basePairForces.update({i: basePairForce()})
        self.forces = basePairForces

    def defineInteraction(self):
        pair_definition = self.nucleic.pair_definition[self.nucleic.pair_definition['nucleic'] == self.nucleic.na_type]
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index
        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'])
        is_rna = atoms['resname'].isin(_rnaResidues)


        contact_list = np.loadtxt(self.contact_list)
        for each_contact_pair in contact_list:
            D1 = atoms[(atoms['resSeq'] == int(each_contact_pair[0])) & is_rna].copy()
            A1 = atoms[(atoms['resSeq'] == int(each_contact_pair[1])) & is_rna].copy()

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