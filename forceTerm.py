# 2020/Oct/11
# No need to add object like class Force(object) in Python 3

import simtk.openmm.app
import simtk.openmm
import simtk.unit as unit

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

    def __init__(self, nucleic, force_group=7):
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