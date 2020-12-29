import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, excepthook=True)

# Total # of particles
numParticles = 8000

# number of divisions in the nxnxn mpm grid
gridDim = 32

# Width of one dimension of a grid node
dx = 1.0 / gridDim

# Inversion of dx, equivalent to # of dimension of node
dxInverse = 1.0 / dx

# Time in one substep
dt = 0.0001

# Possible bug having to do with volume computation here
# Particle volume should be approximately 1/8th the volume of one cube;
# Assumes that ratio of particles and volume of box has been computed above correctly
particleVolume = (1.0/8.0) * (dx ** 3.0)

# Mass per unit volume
particleDensity = 1.0

# Assume uniform density and starting volume for every particle
particleMass = particleVolume * particleDensity

# Strength of the elasticity of this substance; larger values result in more rigid materials
youngsMod = 1000.0

# Between 0.0 and 0.5; the closer to 0.5, the more incompressible
poissonRatio = 0.4

# Lame parameters
# Shear term
mu0 = youngsMod / (2.0 * (1.0 + poissonRatio))

# Dilational term
lambda0 = (youngsMod * poissonRatio) / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio))







