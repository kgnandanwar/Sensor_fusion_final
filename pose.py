import numpy as np

def disambiguate_pose(Rs, Cs, points_3D_sets):
    best_i = 0
    bestValid = 0
    for i, (r, c, points_3D_set) in enumerate(zip(Rs, Cs, points_3D_sets)):
        numValid = 0
        c = c.reshape(-1) 
        r3 = r[2, :]
        for x in points_3D_set:
            view_1 = (x - c)[2]
            view_2 = np.dot(x - c, r3)
            if view_1 > 0 and view_2 > 0:
                # Both cameras looks towards this point
                numValid += 1
        if numValid > bestValid:
            bestValid = numValid
            best_i = i
        print(numValid)
    return Rs[best_i], Cs[best_i], points_3D_sets[best_i]