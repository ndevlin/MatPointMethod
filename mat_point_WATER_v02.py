# Material Point Method Simulator written by Nathan Devlin
# Referenced materials from Professor Chenfanfu Jiang and Xuan Li and Joshuah Wolper

import taichi as ti

# Use GPU for fast, near-real-time simulation,
# use CPU for writing 3D data to .ply files
ti.init(arch=ti.gpu, excepthook=True)

# Total # of particles
numParticles = 128000

# number of divisions in the nxnxn mpm grid
gridDimInt = 50

gridDimFloat = float(gridDimInt)

# Time in one substep; here dt = 1/1000th of 1 frame
dt = 0.00041666

# Strength of the elasticity of this substance; larger values result in more rigid materials
youngsMod = 2.0

# Between 0.0 and 0.5; the closer to 0.5, the more incompressible
poissonRatio = 0.48

bulkModulus = 2.15

gamma = 7.0

# Mass per unit volume
particleDensity = 1.0

# By default, the full simulation area is 1 cubic meter
simulationScale = 1.0

# # of frames per second
frameRate = 24.0

# % of 1 frame that a substep represents; smaller equals a higher temporal resolution
substepSize = 0.05

# Determines the dimensions of the 2 cubes
cube1Width = 0.3
#cube2Width = 0.2

# The time elapsed during one substep
substepTime = (1.0 / frameRate) * substepSize

substepsPerFrame = int(1.0 / substepSize)

# Width of one dimension of a grid node
dx = 1.0 / gridDimFloat

# increase volume to allow overlap of particle volume regions
volumeMultiplier = 5.0

# particle volume equals total volume of shapes divided by number of particles composing them
# This should equate to roughly 8 particles per occupied grid node
particleVolume = volumeMultiplier * (cube1Width ** 3) / float(numParticles)#+ (cube2Width ** 3)) / float(numParticles)

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

# For display purposes
position2D = ti.Vector.field(2, dtype=float, shape=numParticles)

velocity = ti.Vector.field(3, dtype=float, shape=numParticles)

# Affine velocity field to account for particle angular momentum
C = ti.Matrix.field(3, 3, dtype=float, shape=numParticles)

# Deformation gradient matrices
#F = ti.Matrix.field(3, 3, dtype=float, shape=numParticles)

# Plastic deformation factor
Jp = ti.field(dtype=float, shape=numParticles)


# Grid data
# Grid node momentum/velocity
# Note that this field stores momentum first then updates to velocity to save space
gridVelocity = ti.Vector.field(3, dtype=float, shape=(gridDimInt, gridDimInt, gridDimInt))

# Grid node mass
gridMass = ti.field(dtype=float, shape=(gridDimInt, gridDimInt, gridDimInt))

# Data structure to improve speed of transfer of data from taichi to python scope
particleOutput = ti.field(dtype=float, shape=numParticles * 3)

# External force acting on system
gravity = ti.Vector.field(3, dtype=float, shape=())

gravity[None] = [0, -9.8, 0]


# Move particle position data from position[] to particleOutput[] to
# facilitate moving from taichi to python scope for output to a file
@ti.kernel
def processParticleArray():
    for i in range(numParticles * 3):
        particleOutput[(i // 3) * 3] = position[i // 3].x
        particleOutput[(i // 3) * 3 + 1] = position[i // 3].y
        particleOutput[(i // 3) * 3 + 2] = position[i // 3].z


# Output position data to a file
@ti.pyfunc
def writePly(frameNum):
    fileObj = open(str(frameNum) + ".ply", "w")

    fileObj.write("ply\n")
    fileObj.write("format ascii 1.0\n")
    fileObj.write("element vertex " + str(numParticles) + "\n")
    fileObj.write("property float x\n")
    fileObj.write("property float y\n")
    fileObj.write("property float z\n")
    fileObj.write("end_header\n")

    processParticleArray()

    listToPrint = [None] * (numParticles * 3)
    listToPrint = particleOutput

    # Positions to strings
    for i in range(0, numParticles * 3, 3):
        printStr = ""
        printStr += str(listToPrint[i]) + " "
        printStr += str(listToPrint[i + 1]) + " "
        printStr += str(listToPrint[i + 2]) + "\n"
        # write to the file
        fileObj.write(printStr)

    fileObj.close()


# Populate the data structures
# Here we create two cubes give them initial positions, and set all other values to 0
@ti.kernel
def setUp():
    # Create 2 cubes
    particlesPerCube = numParticles #// 2

    # Set the position of the bottom-left corner of the cube
    cube1Base[None] = [0.25, 0.05, 0.5]
    #cube2Base[None] = [0.35, 0.05, 0.5]

    # First Cube
    # Set initial positions by allocating positions randomly within bounds of cube
    for i in range(particlesPerCube):
        position[i] = [ti.random() * cube1Width + cube1Base[None][0],
                       ti.random() * cube1Width + cube1Base[None][1],
                       ti.random() * cube1Width + cube1Base[None][2]]

        #Set intial velocities, angular momentum, Deformation gradient and Plastic deformation to 0
        velocity[i] = [0.0, 0.0, 0.0]
        #F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 3, 3)

    '''
    # Second Cube
    for i in range(particlesPerCube, numParticles):
        position[i] = [ti.random() * cube2Width + cube2Base[None][0],
                       ti.random() * cube2Width + cube2Base[None][1],
                       ti.random() * cube2Width + cube2Base[None][2]]

        # Set intial velocities, angular momentum, Deformation gradient and Plastic deformation to 0
        velocity[i] = [0.0, 0.0, 0.0]
        #F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, 3, 3)
    '''


# Go through one iteration of the simulation
@ti.pyfunc
def substep():
    clearGrid()
    particleToGrid()
    checkBoundaries()
    gridToParticle()


# Set velocities and masses of the grid nodes to 0
@ti.kernel
def clearGrid():
    for i, j, k in gridMass:
        gridMass[i, j, k] = 0.00000001
        gridVelocity[i, j, k] = [0.0, 0.0, 0.0]


# Transfer information about each particle to the corresponding grid nodes
@ti.kernel
def particleToGrid():
    for particle in position:
        # For this particle, compute it's base index, the left/bottom-most
        # node of the 3x3x3 nodes that affect this particle
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


        '''

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
        kirchoffStress = 2 * mu * (F[particle] - (U @ Vtrans)) @ Ftrans + identity * lam * J * (J - 1)
        # @ = Matrix Product in taichi

        '''

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
                       gridDimFloat * weights[i][0] * weights[j][1] * dWeights[k][2]]

            # force contribution of this particle is proportional to its volume, elasticity, and weighting
            #force = -1.0 * particleVolume * kirchoffStress @ dWeight


            force = -1.0 * particleVolume * (-1.0 * bulkModulus * ((1.0 / (Jp[particle] ** gamma)) - 1.0)) * dWeight * Jp[particle]




            # current momentum contribution of this particle to this node equals particle mass times weighting
            gridVelocity[base + offset] += particleMass * weight * (velocity[particle] + C[particle] @ nodePos)

            # mass contribution of this particle to this node equals particle mass times weighting
            gridMass[base + offset] += weight * particleMass

            # momentum equals force * time step; add the computed force to this node's momentum
            gridVelocity[base + offset] += force * dt


# Stop particles from going out of the bounds of the simulation
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

            # Number of grid nodes to use as a border
            boundarySize = 4

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


# Transfer grid node information back to the particles
@ti.kernel
def gridToParticle():
    for particle in position:
        # Base node of 27 nodes around this particle
        base = int(position[particle] * gridDimInt - [0.5, 0.5, 0.5])

        # Distance from base node to particle
        relPosition = position[particle] * gridDimInt - float(base)

        # Quadratic kernels for weighting influence of nearby grid nodes
        weights = [[0.5, 0.5, 0.5] * ([1.5, 1.5, 1.5] - relPosition) * ([1.5, 1.5, 1.5] - relPosition),
                   [0.75, 0.75, 0.75] - (relPosition - [1.0, 1.0, 1.0]) * (relPosition - [1.0, 1.0, 1.0]),
                   [0.5, 0.5, 0.5] * (relPosition - [0.5, 0.5, 0.5]) * (relPosition - [0.5, 0.5, 0.5])]

        # Gradient of interpolation weights
        dWeights = [relPosition - [1.5, 1.5, 1.5],
                    -2.0 * (relPosition - [1.0, 1.0, 1.0]),
                    relPosition - [0.5, 0.5, 0.5]]

        # Create new velocity, affine velocity, and deformation matrix at zero
        newVelocity = ti.Vector.zero(float, 3)
        newC = ti.Matrix.zero(float, 3, 3)
        #newF = ti.Matrix.zero(float, 3, 3)

        velocitySum = 0.0

        # [i, j, k] = which of the 3x3x3 neighbor nodes we are on
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            # position of the current node base relative to this particle
            currNodePos = ti.Vector([float(i), float(j), float(k)]) - relPosition

            # Store velocity currently in this grid node
            currGridVelocity = gridVelocity[base + ti.Vector([i, j, k])]

            # weight value of this node
            weight = weights[i][0] * weights[j][1] * weights[k][2]

            dWeight = ti.Vector.zero(float, 3)
            dWeight = [gridDimFloat * dWeights[i][0] * weights[j][1] * weights[k][2],
                       gridDimFloat * weights[i][0] * dWeights[j][1] * weights[k][2],
                       gridDimFloat * weights[i][0] * weights[j][1] * dWeights[k][2]]

            # new velocity added to this particle = the velocity of the current node * its weighting
            newVelocity += weight * currGridVelocity
            # new affine velocity
            newC += 4.0 * gridDimFloat * weight * currGridVelocity.outer_product(currNodePos)
            # new deformation matrix
            #newF += currGridVelocity.outer_product(dWeight)


            velocitySum += currGridVelocity.dot(dWeight)


        # Assign the computed velocity to this particle
        velocity[particle] = newVelocity
        C[particle] = newC

        # position += the velocity of this particle * the time step
        position[particle] += dt * velocity[particle]

        # Update deformation matrix
        #F[particle] = (ti.Matrix.identity(float, 3) + (dt * newF)) @ F[particle]


        Jp[particle] = (1.0 + dt * velocitySum) * Jp[particle]



# Populate a 2D position field for visualization
@ti.kernel
def from3Dto2D():
    for i in range(numParticles):
        position2D[i] = [position[i].x, position[i].y]


# Begin simulation main code

print("Begin MPM Simulation")

gui = ti.GUI("MPM Simulation", res=720, background_color=0x000000)

setUp()

writePly(0)

from3Dto2D()

gui.circles(position2D.to_numpy(), radius = 2.0, color=0x990000)
gui.show()

numFrames = 200

for frame in range(numFrames):
    print("\nFrame " + str(frame))

    for step in range(substepsPerFrame):
        #print("Step " + str(step))
        substep()

    # Note: Disable writePly for near-realtime GPU processing
    #writePly(frame)

    from3Dto2D()

    gui.circles(position2D.to_numpy(), radius = 2.0, color=0x990000)

    gui.show()

