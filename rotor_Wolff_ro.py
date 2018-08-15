from MMTK import *
import sys
sys.path.append("/home/x3lou/MMTK-PIMC")
from MMTK.ForceFields.ForceField import CompoundForceField
from MMTK.ForceFields.ForceFieldTest import gradientTest
from MMTK import Features
from RotOnly_WolffIntegrator import RotOnlyWolff_PINormalModeIntegrator, RotOnlyWolff_PILangevinNormalModeIntegrator
from dipoleFF import dipoleForceField
from MMTK.Environment import PathIntegrals
from MMTK.NormalModes import VibrationalModes
from MMTK.Trajectory import Trajectory, TrajectoryOutput, \
                            RestartTrajectoryOutput, StandardLogOutput, \
                            trajectoryInfo
from sys import argv,exit
from Scientific.Statistics import mean, standardDeviation
from MMTK.Minimization import ConjugateGradientMinimizer
from Scientific import N
from numpy import zeros,cos,sin,sqrt,pi, dot, asarray, sign, arctan
import subprocess
import re

############################
# set the number of quantum beads
############################
label="Wolff-test"
outdir="/work/x3lou/MMTK/HFRotations/"
# Parameters
densname = argv[1]
temperature = float(densname[densname.find("T")+1:densname.find("P")])*Units.K  # temperature
P = int(densname[densname.find("P")+1:densname.find(".dat")])                    # number of beads

## lattice spacing
lattice_spacing=10.05*Units.Ang

ndensn = int(densname[densname.find("n")+1:densname.find("e")])
ntens = int(densname[densname.find("e")+1:densname.find("T")])
ndens=ndensn*(10**ntens)
print ndens

rho=zeros(ndens,float)
roteng=zeros(ndens,float)

Rot_Step = 1.0 #Rot Step, no need for this argv, just leave it here
Rot_Skip = 1 #Rot Skip Step, no need for this argv, just leave it here

universe = InfiniteUniverse()
# nanometers

universe.addObject(PathIntegrals(temperature))

## number of molecules
nmolecules = int(argv[2])

## They will initially be aligned along the z-axis
for i in range(nmolecules):
	universe.addObject(Molecule('hf', position = Vector(i*lattice_spacing, 0., 0.)))

for atom in universe.atomList():
	atom.setNumberOfBeads(P)


print "ATOMS"
print  universe.atomList()
natoms = len(universe.atomList())
#print universe.atomList()[0].mass()
#print universe.atomList()[1].mass()

ff=[]
##################################################
############## DIPOLE POTENTIAL #################
##################################################
for i in range(nmolecules):
        for j in range(i+1,nmolecules):
                ff.append(dipoleForceField(universe.objectList()[i].atomList()[0],universe.objectList()[i].atomList()[1],
                                           universe.objectList()[j].atomList()[0],universe.objectList()[j].atomList()[1]))

universe.setForceField(CompoundForceField(*ff))

#print "last energy values"
print universe.energyTerms()
print "last energy values"
print universe.energyEvaluator().CEvaluator().last_energy_values
#raise()

universe.writeToFile("u.pdb")
universe.initializeVelocitiesToTemperature(temperature)

densfile=open(densname,"r")
for i in range(ndens):
        dummy=densfile.readline().split()
        rho[i]=float(dummy[1])
        roteng[i]=float(dummy[2])*0.0083144621 #K to kJ/mol [MMTK Units of Energy]
densfile.close()

dt = 1.0*Units.fs

# Initialize velocities
universe.initializeVelocitiesToTemperature(temperature)

# USE THE FRICTION PARAMETER FROM BEFORE
friction = 0.0
integrator = RotOnlyWolff_PILangevinNormalModeIntegrator(universe, delta_t=dt, centroid_friction = friction, densmat=rho,rotengmat=roteng, rotstep=float(Rot_Step), rotskipstep=int(Rot_Skip))

integrator(steps=5000, actions = [ TrajectoryOutput(None,('configuration','time'), 0, None, 100)] )

RunSteps = 50.0*Units.ps/dt
SkipSteps = 50.0*Units.fs/dt

trajectory = Trajectory(universe, outdir+str(nmolecules)+"HF-P"+str(P)+"_"+label+".nc", "w", "A simple test case")
Nblocks=1

############################## BEGIN ROTATION SIMULATION ##############################

# RUN PIMD WITH PIMC ROTATION INCLUDED
print "We're going to run the Langevin integrator for ", RunSteps/SkipSteps, "independent steps of PIMD"
integrator(steps=RunSteps,
           # Remove global translation every 50 steps.
	   actions = [
		   TrajectoryOutput(trajectory, ("time", "thermodynamic", "energy",
						 "configuration", "auxiliary"),
                                    0, None, SkipSteps)])

npoints = len(trajectory)
universe = trajectory.universe
natoms = universe.numberOfAtoms()
time=trajectory.time
np = universe.numberOfPoints()
P = np/natoms
gradientTest(universe)
trajectory.close()

