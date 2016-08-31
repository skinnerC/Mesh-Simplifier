import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.spatial import Delaunay, ConvexHull
import triangle
import networkx as nx
from deap import creator, base, tools, algorithms
from collections import Counter
    
def parseVertices(org_mesh):
    # Retrieve vertex data
    x = org_mesh.x.reshape((-1, 1))
    y = org_mesh.y.reshape((-1, 1))
    z = org_mesh.z.reshape((-1, 1))
    vertices = np.hstack((x,y,z))
    
    # Remove dupicate vertices
    # Perform lex sort and get sorted data
    sorted_idx = np.lexsort(vertices.T)
    sorted_data =  vertices[sorted_idx,:]
    # Get unique row mask
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    # Get unique rows
    vertices = sorted_data[row_mask]
    
    return vertices

def getMesh(xyz):#getMesh(xyz, xy):
    faces = Delaunay(xyz, qhull_options="QJ").simplices#triangle.delaunay(xy)
    new_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            new_mesh.vectors[i][j] = xyz[f[j],:]
    
    return new_mesh
    
def calcError(new_mesh, x, y, z):
    vertices = parseVertices(new_mesh)
    # Find matching x, y values
    mask = np.in1d(vertices[:,0], x) & np.in1d(vertices[:,1], y)
    #print mask
    # Get indices of matching of values
    indices = []
    for i, j in enumerate(mask):
        if j == True:
            indices += [i] 
    z = z[indices]
    # Subtract new z from old z and sum all differences
    newZ = vertices[:,2].reshape((-1, 1))
    #error = np.subtract(z,newZ)
    for i,j in enumerate(newZ):
        z[i] -= j
    error = np.absolute(z)
    error = error.sum()
    return error

def getImportant(org_mesh, vertices, numImportant):
    areas = org_mesh.areas
    vectors = org_mesh.points
    areas = np.repeat(areas, 3).reshape((-1, 1))
    
    vectors = vectors.reshape((int(areas.shape[0]), 3))
    a = vectors.tolist()
    print a[0]
    a = map(str, a)
    print a[0]
    c = Counter(a)
    imp = c.most_common(numImportant)
    print imp
    
    vertices = map(str, vertices)
    idx = []
    for i in imp:
        try:
            idx += vertices.index(i[0])
        except:
            pass
    
    print idx
    return idx
    

def main(filename):
    np.random.seed(0)
    
    # Parse STL
    org_mesh = mesh.Mesh.from_file(filename)
    vertices = parseVertices(org_mesh)
    x = vertices[:,0].reshape((-1, 1))
    y = vertices[:,1].reshape((-1, 1))
    z = vertices[:,2].reshape((-1, 1))

    #numImportant = 100    
    #getImportant(org_mesh, vertices, numImportant)
    
    NVERT = int(vertices.shape[0]) # vertices in orignial model
    NGEN = 50
    MU = 15
    LAMBDA = 20
    CXPB = 0.7
    MUTPB = 0.2
        
    #print vertices
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
        
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NVERT)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        # Build list of vertices
        indices = []
        for i, j in enumerate(individual):
            if j == 1:
                indices += [i]
        
        points = vertices[indices]#flat_vert[indices]
        
        # Generate mesh from individual
        try:
            new_mesh =  getMesh(points)#(vertices, points)
        except ValueError:
            return (int(vertices.shape[0])*2, int(vertices.shape[0])*2)
        
        return (len(points), calcError(new_mesh, x, y, z))
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5) #0.05
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/NVERT) #0.05
    toolbox.register("select", tools.selNSGA2)
    
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)    
    
    pop = toolbox.population(n=MU)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)
    
    file = open("stats_{0}.txt".format(filename), "w")
    file.write(str(log))
    file.write("\n")
    for ind in pop:
        file.write(str(ind.fitness))
    file.close()
    
    indices = []
    for i, j in enumerate(hof[0]):
        if j == 1:
            indices += [i]
    
    points = vertices[indices]  
    
    new_mesh = getMesh(points)
    new_mesh.update_normals
    new_mesh.update_areas
    new_mesh.save("new_{0}.stl".format(filename))
    
if __name__ == "__main__":
    files = ["Pear.stl"]
    for file in files:
        main(file)