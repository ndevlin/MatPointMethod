import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, excepthook=True)

# Total # of particles
numParticles = 8000

# number of divisions in the nxnxn mpm grid
gridDim = 32

# Time in one substep
dt = 0.0001

# Strength of the elasticity of this substance; larger values result in more rigid materials
youngsMod = 1000.0

# Between 0.0 and 0.5; the closer to 0.5, the more incompressible
poissonRatio = 0.4

# Mass per unit volume
particleDensity = 1.0


# Width of one dimension of a grid node
dx = 1.0 / gridDim

# Inversion of dx, equivalent to # of dimension of node
dxInverse = 1.0 / dx


# Possible bug having to do with volume computation here
# Particle volume should be approximately 1/8th the volume of one cube;
# Assumes that ratio of particles and volume of box has been computed above correctly
particleVolume = (1.0 / 8.0) * (dx ** 3.0)

# Assume uniform density and starting volume for every particle
particleMass = particleVolume * particleDensity

# Lame parameters
# Shear term
mu0 = youngsMod / (2.0 * (1.0 + poissonRatio))

# Dilational term
lambda0 = (youngsMod * poissonRatio) / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio))

# Bottom-left starting corner of each cube
cube1Base = ti.Vector.field(3, dtype=float, shape=())
cube2Base = ti.Vector.field(3, dtype=float, shape=())

# Particle data

position = ti.Vector.field(3, dtype=float, shape=numParticles)

velocity = ti.Vector.field(3, dtype=float, shape=numParticles)

# Affine velocity field to account for particle angular momentum
C = ti.Matrix.field(3, 3, dtype=float, shape=numParticles)

# Deformation gradient matrices
F = ti.Matrix.field(3, 3, dtype=float, shape=numParticles)

# Plastic deformation factor
Jp = ti.field(dtype=float, shape=numParticles)


# Grid data
# Grid node momentum/velocity
gridVelocity = ti.Vector.field(3, dtype=float, shape=(gridDim, gridDim, gridDim))

# Grid node mass
gridMass = ti.field(dtype=float, shape=(gridDim, gridDim, gridDim))

# External force acting on system
gravity = ti.Vector.field(3, dtype=float, shape=())

gravity[None] = [0, -9.8, 0]



@ti.kernel
def setUp():
    # Create 2 cubes

    particlesPerCube = numParticles // 2

    cube1Base[None] = [0.25, 0.5, 0.6]
    cube1Width = 0.2

    cube2Base[None] = [0.35, 0.5, 0.1]
    cube2Width = 0.2

    # First Cube
    # Set initial positions by allocating positions randomly within bounds of cube
    for i in range(particlesPerCube):
        position[i] = [ti.random() * cube1Width + cube2Base[None][0],
                       ti.random() * cube1Width + cube2Base[None][1],
                       ti.random() * cube1Width + cube2Base[None][2]]

        #Set intial velocities, angular momentum, Deformation gradient and Plastic deformation to 0
        velocity[i] = [0.0, 0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 3, 3)

    # Second Cube
    for i in range(particlesPerCube, numParticles):
        position[i] = [ti.random() * cube2Width + cube2Base[None][0],
                       ti.random() * cube2Width + cube2Base[None][1],
                       ti.random() * cube2Width + cube2Base[None][2]]

        # Set intial velocities, angular momentum, Deformation gradient and Plastic deformation to 0
        velocity[i] = [0.0, 0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 3, 3)









print("Begin MPM Simulation")

setUp()
