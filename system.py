# 2020/Oct/11
# No need to add object like class Force(object) in Python 3

import simtk.openmm.app
import simtk.openmm
import simtk.unit as unit
import numpy as np
import force

# List forces
forces = dict(Backbone=force.Backbone)

class System(simtk.openmm.System):
    """ Wrapper of openmm system class, adds some openmm simulation attributes"""

    def __init__(self, nucleic, forcefieldFiles=[f'./openna.xml'], periodicBox=None):
        self.nucleic = nucleic
        self.top = simtk.openmm.app.PDBFile(nucleic.pdb_file).getTopology()
        if self.top.getUnitCellDimensions() is None:
            x = nucleic.atoms[['x', 'y', 'z']]
            d = np.round((x.max() - x.min()) * 2 + 5, -1)
            self.top.setUnitCellDimensions(d)

        self.coord = simtk.openmm.app.PDBFile(nucleic.pdb_file)
        self.forcefield = simtk.openmm.app.ForceField(*forcefieldFiles)
        self._wrapped_system = self.forcefield.createSystem(self.top)
        self.periodicBox = periodicBox
        if periodicBox is not None:
            self._wrapped_system.setDefaultPeriodicBoxVectors(*np.diag(self.periodic_box))
            self.nucleic.periodic = True
        elif periodicBox is None and self.nucleic.periodic == True: # why this line need __getattr__?
            self.nucleic.periodic = False
            print('Periodic boundary conditions not defined, system would be non periodic')
        self.forces = {}
        
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_system, attr)


    def clearForces(self, keepCMMotionRemover=True):
        """ Removes all the forces from the system.
        openMM usually adds a "CMMotionRemover" force to keep the center of mass of the system from drifting."""
        j = 0
        for i, f in enumerate(self.getForces()):
            if keepCMMotionRemover and i == 0 and f.__class__ == simtk.openmm.CMMotionRemover:
                # print('Kept ', f.__class__)
                j += 1
                continue
            else:
                # print('Removed ', f.__class__)
                self.removeForce(j)
        if keepCMMotionRemover == False:
            assert len(self.getForces()) == 0, 'Not all the forces were removed'
        else:
            assert len(self.getForces()) <= 1, 'Not all the forces were removed'

    def addforces(self, verbose=False):
        """ Adds all forces"""
        self.addForce(force)
        self.forces.update({force_name: force})

    def initializeMD(self, temperature=300 * unit.kelvin, platform_name='Reference', damping=2/unit.picosecond, timestep=2*unit.femtoseconds):
        """Starts a simple simulation using the selected system"""
        self.integrator = simtk.openmm.LangevinIntegrator(temperature, damping, timestep)
        self.platform = simtk.openmm.Platform.getPlatformByName(platform_name)
        self.simulation = simtk.openmm.app.Simulation(self.top, self._wrapped_system, self.integrator, self.platform)
        self.simulation.context.setPositions(self.coord.positions)
        return self.simulation

    def setPositions(self, coords=None):
        """Sets the particle positions in the simulation"""
        # Initialize trial MD if not setup
        try:
            self.simulation
        except AttributeError:
            self.initializeMD()

        # Set up coords for MD
        if coords is None:
            self.simulation.context.setPositions(self.coord.positions)
        else:
            self.simulation.context.setPositions(coords)

    def getPotentialEnergy(self, coords=None, energy_unit=unit.kilojoule_per_mole):
        """Returns the potential energy of the current state of the system (default unit KJ/mol)"""
        # Initialize trial MD if not setup
        try:
            self.simulation
        except AttributeError:
            self.initializeMD()
        self.setPositions(coords)
        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(energy_unit)
        return energy

