import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, excepthook=True)

# Total # of particles
numParticles = 8000

# number of divisions in the nxnxn mpm grid
gridDimInt = 32

gridDimFloat = float(gridDimInt)

# Time in one substep; here dt = 1/1000th of 1 frame
dt = 0.000041666

# Strength of the elasticity of this substance; larger values result in more rigid materials
youngsMod = 1000.0

# Between 0.0 and 0.5; the closer to 0.5, the more incompressible
poissonRatio = 0.4

# Mass per unit volume
particleDensity = 1.0

# By default, the full simulation area is 1 cubic meter
simulationScale = 1.0

# # of frames per second
frameRate = 24.0

# % of 1 frame that a substep represents; smaller equals a higher resolution
substepSize = 0.05

# The time elapsed during one substep
substepTime = (1.0 / frameRate) * substepSize

substepsPerFrame = int(1.0 / substepSize)

# Width of one dimension of a grid node
dx = 1.0 / gridDimFloat

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
# Note that this field stores momentum first then updates to velocity to save space
gridVelocity = ti.Vector.field(3, dtype=float, shape=(gridDimInt, gridDimInt, gridDimInt))

# Grid node mass
gridMass = ti.field(dtype=float, shape=(gridDimInt, gridDimInt, gridDimInt))

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


@ti.kernel
def clearGrid():
    # Set Grid values to 0
    for i, j, k in gridMass:
        gridMass[i, j, k] = 0.00000001
        gridVelocity[i, j, k] = [0.0, 0.0, 0.0]


@ti.pyfunc
def substep():
    clearGrid()
    particleToGrid()
    checkBoundaries()


@ti.kernel
def checkBoundaries():
    for i, j, k in gridMass:
        if gridMass[i, j, k] <= 0.0:
            gridMass[i, j, k] = 0.00000001
        if gridMass[i, j, k] > 0.00000002:
            # gridVelocity currently stores momentum; divide by mass to give velocity
            gridVelocity[i, j, k] /= gridMass[i, j, k]

            # increase velocity according to acceleration over the amount of time of one substep
            gridVelocity[i, j, k] += gravity[None] * substepTime * simulationScale

            boundarySize = 5

            # Wall collisions
            # Left wall
            if i < boundarySize and gridVelocity[i, j, k].x < 0.0:
                gridVelocity[i, j, k].x = 0.0
            # Right wall
            if i > gridDimInt - boundarySize and gridVelocity[i, j, k].x > 0.0:
                gridVelocity[i, j, k].x = 0.0
            # Floor
            if j < boundarySize and gridVelocity[i, j, k].y < 0.0:
                gridVelocity[i, j, k].y = 0.0
            # Ceiling
            if j > gridDimInt - boundarySize and gridVelocity[i, j, k].y > 0.0:
                gridVelocity[i, j, k].y = 0.0
            # Front wall
            if k < boundarySize and gridVelocity[i, j, k].z < 0.0:
                gridVelocity[i, j, k].z = 0.0
            # Back wall
            if k > gridDimInt - boundarySize and gridVelocity[i, j, k].z > 0.0:
                gridVelocity[i, j, k].z = 0.0


@ti.kernel
def particleToGrid():
    # Calculate particle values (Particle to Grid)
    for particle in position:
        # for this particle, compute it's base index
        # left/bottom-most node of the 3x3x3 nodes that affect this particle

        # Check here for syntax error with int casting and with 0.5 vector
        base = int(position[particle] * gridDimFloat - [0.5, 0.5, 0.5])
        # distance vector from particle position to base node position
        relPosition = position[particle] * gridDimFloat - float(base)

        # Quadratic kernels for weighting influence of nearby grid nodes
        weights = [[0.5, 0.5, 0.5] * ([1.5, 1.5, 1.5] - relPosition) * ([1.5, 1.5, 1.5] - relPosition),
                   [0.75, 0.75, 0.75] - (relPosition - [1.0, 1.0, 1.0]) * (relPosition - [1.0, 1.0, 1.0]),
                   [0.5, 0.5, 0.5] * (relPosition - [0.5, 0.5, 0.5]) * (relPosition - [0.5, 0.5, 0.5])]

        # Gradient of interpolation weights
        dWeights = [relPosition - [1.5, 1.5, 1.5],
                    -2.0 * (relPosition - [1.0, 1.0, 1.0]),
                    relPosition - [0.5, 0.5, 0.5]]

        # mu and lambda remain unchanged under normal elastic conditions
        mu = mu0
        lam = lambda0

        # Polar Singular Value Decomposition
        U, sigma, V = ti.svd(F[particle])
        J = 1.0

        # Modify deformation gradient
        for i in ti.static(range(3)):
            J *= sigma[i, i]

        # Compute Kirchoff stress for elasticity
        Ftrans = F[particle].transpose()
        Vtrans = V.transpose()
        identity = ti.Matrix.identity(float, 3)
        # Compute Kirchoff Stress for elasticity
        kirchoffStress = 2 * mu * (F[particle] - (U @ Vtrans)) @ Ftrans + identity * lam * J * (J - 1)
        # @ = Matrix Product in taichi

        # Update particle to grid velocity, mass and force
        # Iterate throught the 27 grid node neighbors
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            # which node we are in within the 3x3x3 affecting nodes
            offset = ti.Vector([i, j, k])

            # position of the corner of this node relative to particle
            nodePos = (float(offset) - relPosition) * dx

            # weight value of this node
            weight = weights[i][0] * weights[j][1] * weights[k][2]

            dWeight = ti.Vector.zero(float, 3)
            dWeight = [gridDimFloat * dWeights[i][0] * weights[j][1] * weights[k][2],
                       gridDimFloat * weights[i][0] * dWeights[j][1] * weights[k][2],
                       gridDimFloat * dWeights[i][0] * weights[j][1] * dWeights[k][2]]

            # force contribution of this particle is proportional to its volume, elasticity, and weighting
            force = -1.0 * particleVolume * kirchoffStress @ dWeight

            # current momentum contribution of this particle to this node equals particle mass times weighting
            gridVelocity[base + offset] += particleMass * weight * (velocity[particle] + C[particle] @ nodePos)

            # mass contribution of this particle to this node equals particle mass times weighting
            gridMass[base + offset] += weight * particleMass

            # momentum equals force * time step; add the computed force to this particle's momentum
            gridVelocity[base + offset] += force * dt



print("Begin MPM Simulation")

setUp()

numFrames = 500

for frame in range(numFrames):
    print("\nFrame " + str(frame))

    for step in range(substepsPerFrame):
        print("Step " + str(step))
        substep()

