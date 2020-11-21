class CrossStacking(Force):
    def __init__(self, dna, force_group=11, OpenCLPatch=True):
        self.force_group = force_group
        super().__init__(dna, OpenCLPatch)

    def reset(self):
        def crossStackingForce(parametersOnDonor=False):
            if self.OpenCLPatch:
                # Divide the energy over two to make the interaction symetric
                patch_string = '/2'
            else:
                patch_string = ''
            crossForce = simtk.openmm.CustomHbondForce(f'''energy{patch_string};temp=dtCS/rng_CS;
                                                        energy   = fdt3*fdtCS*attr;
                                                        attr     = epsilon*(1-exp(-alpha*dr))^2*step(dr)-epsilon;
                                                        fdt3     = max(f1*pair0t3,pair1t3);
                                                        fdtCS    = max(f2*pair0tCS,pair1tCS);
                                                        pair0t3  = step(pi+dt3)*step(pi-dt3);
                                                        pair0tCS = step(pi+dtCS)*step(pi-dtCS);
                                                        pair1t3  = step(pi/2+dt3)*step(pi/2-dt3);
                                                        pair1tCS = step(pi/2+dtCS)*step(pi/2-dtCS);
                                                        f1       = 1-cos(dt3)^2;
                                                        f2       = 1-cos(dtCS)^2;
                                                        dr       = distance(d1,a3)-sigma;
                                                        dt3      = rng_BP*(t3-t03);
                                                        dtCS     = rng_CS*(tCS-t0CS);
                                                        tCS      = angle(d2,d1,a3);
                                                        t3       = acos(cost3lim);
                                                        cost3lim = min(max(cost3,-0.99),0.99);
                                                        cost3    = sin(t1)*sin(t2)*cos(phi)-cos(t1)*cos(t2);
                                                        t1       = angle(d2,d1,a1);
                                                        t2       = angle(d1,a1,a2);
                                                        phi      = dihedral(d2,d1,a1,a2);''')
            if self.periodic:
                crossForce.setNonbondedMethod(crossForce.CutoffPeriodic)
            else:
                crossForce.setNonbondedMethod(crossForce.CutoffNonPeriodic)
            crossForce.setCutoffDistance(1.8)  # Paper
            parameters = ['t03', 't0CS', 'rng_CS', 'rng_BP', 'epsilon', 'alpha', 'sigma']
            for p in parameters:
                if parametersOnDonor:
                    crossForce.addPerDonorParameter(p)
                else:
                    crossForce.addPerAcceptorParameter(p)
            crossForce.addGlobalParameter('pi', np.pi)
            crossForce.setForceGroup(self.force_group)
            return crossForce

        crossStackingForces = {}
        for base in ['A', 'T', 'G', 'C']:
            crossStackingForces.update({base: (crossStackingForce(), crossStackingForce())})
        self.crossStackingForces = crossStackingForces

    def defineInteraction(self):
        atoms = self.dna.atoms.copy()
        atoms['index'] = atoms.index
        atoms.index = zip(atoms['chainID'], atoms['resSeq'], atoms['name'].replace(['A', 'C', 'T', 'G'], 'B'))
        is_dna = atoms['resname'].isin(_dnaResidues)
        bases = atoms[atoms['name'].isin(['A', 'T', 'G', 'C']) & is_dna]
        D1 = bases
        D2 = atoms.reindex([(c, r, 'S') for c, r, n in bases.index])
        D3 = atoms.reindex([(c, r + 1, 'B') for c, r, n in bases.index])
        A1 = D1
        A2 = D2
        A3 = atoms.reindex([(c, r - 1, 'B') for c, r, n in bases.index])

        # Select only bases where the other atoms exist
        D2.index = D1.index
        D3.index = D1.index
        temp = pandas.concat([D1, D2, D3], axis=1, keys=['D1', 'D2', 'D3'])
        sel = temp[temp['D3', 'name'].isin(['A', 'T', 'G', 'C']) &  # D3 must be a base
                   temp['D2', 'name'].isin(['S']) &  # D2 must be a sugar
                   (temp['D3', 'chainID'] == temp['D1', 'chainID']) &  # D3 must be in the same chain
                   (temp['D2', 'chainID'] == temp['D1', 'chainID'])].index  # D2 must be in the same chain
        D1 = atoms.reindex(sel)
        D2 = atoms.reindex([(c, r, 'S') for c, r, n in sel])
        D3 = atoms.reindex([(c, r + 1, 'B') for c, r, n in sel])

        # Aceptors
        A2.index = A1.index
        A3.index = A1.index
        temp = pandas.concat([A1, A2, A3], axis=1, keys=['A1', 'A2', 'A3'])
        sel = temp[temp['A3', 'name'].isin(['A', 'T', 'G', 'C']) &  # A3 must be a base
                   temp['A2', 'name'].isin(['S']) &  # A2 must be a sugar
                   (temp['A3', 'chainID'] == temp['A1', 'chainID']) &  # A3 must be in the same chain
                   (temp['A2', 'chainID'] == temp['A1', 'chainID'])].index  # A2 must be in the same chain
        A1 = atoms.reindex(sel)
        A2 = atoms.reindex([(c, r, 'S') for c, r, n in sel])
        A3 = atoms.reindex([(c, r - 1, 'B') for c, r, n in sel])

        # Parameters
        cross_definition = self.dna.cross_definition[self.dna.cross_definition['DNA'] == self.dna.DNAtype].copy()
        i = [a for a in zip(cross_definition['Base_d1'], cross_definition['Base_a1'], cross_definition['Base_a3'])]
        cross_definition.index = i

        donors = {i: [] for i in ['A', 'T', 'G', 'C']}
        for donator, donator2, d1, d2, d3 in zip(D1.itertuples(), D3.itertuples(), D1['index'], D2['index'],
                                                 D3['index']):
            # if d1!=4:
            #    continue
            d1t = donator.name
            d3t = donator2.name
            c1, c2 = self.crossStackingForces[d1t]
            a1t = _complement[d1t]
            # print(d1, d2, d3)
            param = cross_definition.loc[[(a1t, d1t, d3t)]].squeeze()
            # parameters=[param1['t03']*af,param1['T0CS_1']*af,param1['rng_cs1'],param1['rng_bp'],param1['eps_cs1']*ef,param1['alpha_cs1']/df,param1['Sigma_1']*df]
            parameters = [param['t03'] * _af,
                          param['T0CS_2'] * _af,
                          param['rng_cs2'],
                          param['rng_bp'],
                          param['eps_cs2'] * _ef,
                          param['alpha_cs2'] / _df,
                          param['Sigma_2'] * _df]
            # print(param)
            c1.addDonor(d1, d2, d3)
            c2.addAcceptor(d1, d2, d3, parameters)
            # print("Donor", d1t, d1, d2, d3)
            donors[d1t] += [d1]

        aceptors = {i: [] for i in ['A', 'T', 'G', 'C']}
        for aceptor, aceptor2, a1, a2, a3 in zip(A1.itertuples(), A3.itertuples(), A1['index'], A2['index'],
                                                 A3['index']):
            # if a1!=186:
            #    continue
            a1t = aceptor.name
            a3t = aceptor2.name
            c1, c2 = self.crossStackingForces[_complement[a1t]]
            d1t = _complement[a1t]
            param = cross_definition.loc[[(d1t, a1t, a3t)]].squeeze()
            # print(param)
            # print(a1, a2, a3)
            parameters = [param['t03'] * _af,
                          param['T0CS_1'] * _af,
                          param['rng_cs1'],
                          param['rng_bp'],
                          param['eps_cs1'] * _ef,
                          param['alpha_cs1'] / _df,
                          param['Sigma_1'] * _df]
            # parameters=[param1['t03']*af,param1['T0CS_2']*af,param1['rng_cs2'],param1['rng_bp'],param1['eps_cs2']*ef,param1['alpha_cs2']/df,param1['Sigma_2']*df]
            c1.addAcceptor(a1, a2, a3, parameters)
            c2.addDonor(a1, a2, a3)
            # print("Aceptor", a1t, a1, a2, a3)
            aceptors[_complement[a1t]] += [a1]

        # Exclusions
        for base in ['A', 'T', 'G', 'C']:
            c1, c2 = self.crossStackingForces[base]
            for ii, i in enumerate(donors[base]):
                for jj, j in enumerate(aceptors[base]):
                    # The sequence exclusion was reduced to two residues
                    # since the maximum number of exclusions in OpenCL is 4.
                    # In the original 3SPN2 it was 3 residues (6 to 9)
                    # This change has a small effect in B-DNA and curved B-DNA
                    # The second change is to make the interaction symetric and dividing the energy over 2
                    # This also reduces the number of exclusions in the force
                    if self.OpenCLPatch:
                        maxn = 6
                    else:
                        maxn = 9
                    if (self.dna.atoms.at[i, 'chainID'] == self.dna.atoms.at[j, 'chainID'] and abs(i - j) <= maxn) or \
                            (not self.OpenCLPatch and i > j):
                        c1.addExclusion(ii, jj)
                        c2.addExclusion(jj, ii)

    def addForce(self, system):
        for c1, c2 in self.crossStackingForces.values():
            system.addForce(c1)
            system.addForce(c2)

    def getForceGroup(self):
        fg = 0
        for c1, c2 in self.crossStackingForces.values():
            fg = c1.getForceGroup()
            break
        for c1, c2 in self.crossStackingForces.values():
            assert fg == c1.getForceGroup()
            assert fg == c2.getForceGroup()
        return fg