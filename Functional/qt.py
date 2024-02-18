import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math
def rotation_equations(w, ls):
        # w = (l0, alpha, beta, gamma)

        l0 = w[0]
        alpha = w[1]
        beta = w[2]
        gamma = w[3]

        if len(ls) == 3:
                l1_est = rotate([l0, 0, 0], alpha, beta, gamma)
                l2_est = rotate([l0 * np.cos(np.pi/3), l0 * np.sin(np.pi/3), 0], alpha, beta, gamma)
                l3_est = rotate([-l0 * np.cos(np.pi/3), l0 * np.sin(np.pi/3), 0], alpha, beta, gamma)

                l1x = ls[0][0]
                l1y = ls[0][1]
                l2x = ls[1][0]
                l2y = ls[1][1]
                l3x = ls[2][0]
                l3y = ls[2][1]
                
                F = np.zeros(4)
                F[0] = l2_est[0]-l3_est[0] - (l2x-l3x)
                F[1] = l2_est[1]-l3_est[1] - (l2y-l3y)
                F[2] = l1_est[0]-l2_est[0] - (l1x-l2x)
                F[3] = l1_est[1]-l2_est[1] - (l1y-l2y)
        else:
                F = [0,0,0,0]
                print("not enough ls")
                
        #F[0] = l1_est[0] - l1x
        #print(l1_est[0], l1x)
        #F[1] = l1_est[1] - l1y
        #F[2] = l2_est[0] - l2x
        #F[3] = l2_est[1] - l2y
        #if len(ls) == 3:
        #    F[4] = l3_est[0] - l3x
        #    F[5] = l3_est[1] - l3y
        return F

def rotate(vector, alpha, beta, gamma):
        cos = np.cos
        sin = np.sin
        Rx = np.matrix([[1, 0, 0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha), cos(alpha)]])
        Ry = np.matrix([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
        Rz = np.matrix([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma), cos(gamma), 0],
                        [0, 0, 1]])

        R = Rz @ Ry @ Rx
        #R = np.matrix([[cos(beta) * cos(gamma), sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma), cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
                #              [cos(beta) * sin(gamma), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma)],
                #              [-sin(beta), sin(alpha) * cos(beta), cos(alpha) * cos(beta)]])
        #print(np.matmul(R, np.transpose(np.matrix(vector))).flatten().tolist()[0])

        return np.matmul(R, np.transpose(np.matrix(vector))).flatten().tolist()[0]

def unrotate(vector, alpha, beta, gamma):
        cos = np.cos
        sin = np.sin
        Rx = np.matrix([[1, 0, 0],
                        [0, cos(alpha), -sin(alpha)],
                        [0, sin(alpha), cos(alpha)]])
        Ry = np.matrix([[cos(beta), 0, sin(beta)],
                        [0, 1, 0],
                        [-sin(beta), 0, cos(beta)]])
        Rz = np.matrix([[cos(gamma), -sin(gamma), 0],
                        [sin(gamma), cos(gamma), 0],
                        [0, 0, 1]])
        R = Rx @ Ry @ Rz
        return np.squeeze(np.asarray(R @ np.transpose(np.matrix(vector))))

for i in range(0,1000):
        hexagon = np.array([[5,0,0], [5 * np.cos(np.pi/3), 5 * np.sin(np.pi/3), 0], [-5 * np.cos(np.pi/3), 5 * np.sin(np.pi/3), 0], [-5,0,0], [-5 * np.cos(np.pi/3), -5 * np.sin(np.pi/3), 0], [5 * np.cos(np.pi/3), -5 * np.sin(np.pi/3), 0], [5,0,0]])
        alpha = np.random.uniform(0, np.pi/2)
        beta = np.random.uniform(0, np.pi/2)
        gamma = np.random.uniform(0, np.pi/2)
        rotated_hexagon = np.array([rotate(vector, alpha, beta, gamma) for vector in hexagon])
        #print(f"Original hexagon: {hexagon}")
        #print(f"Angles are {alpha}, {beta}, {gamma}")
        #print(f"Rotated hexagon: {rotated_hexagon}")

        ls = [[rotated_hexagon[0][0], rotated_hexagon[0][1]], [rotated_hexagon[1][0], rotated_hexagon[1][1]], [rotated_hexagon[2][0], rotated_hexagon[2][1]]]

        params_init = [4.5, np.pi/4, np.pi/4, np.pi/4]
        #print(l0, l1)

        params = fsolve(rotation_equations, x0=params_init, args=ls)
        #print(f"parameters found: {params}")

        param_hexagon = np.array([[params[0],0,0], [params[0] * np.cos(np.pi/3), params[0] * np.sin(np.pi/3), 0], [-params[0] * np.cos(np.pi/3), params[0] * np.sin(np.pi/3), 0], [-params[0],0,0], [-params[0] * np.cos(np.pi/3), -params[0] * np.sin(np.pi/3), 0], [params[0] * np.cos(np.pi/3), -params[0] * np.sin(np.pi/3), 0], [params[0],0,0]])
        param_rotated_hexagon = np.array([rotate(vector, params[1], params[2], params[3]) for vector in param_hexagon])
        #print(f"param rotated: {param_rotated_hexagon}")
        if math.isclose(alpha, params[1], rel_tol=1E-6) and math.isclose(beta, params[2], rel_tol=1E-6) and math.isclose(gamma, params[3], rel_tol=1E-6):
                print("Pass")
                pass
        else:
                print("Fail")
                print(f"Angles are {alpha}, {beta}, {gamma}")
                print(f"parameters found: {params}")
                fig = plt.figure()
                ax= plt.axes(projection="3d")
                hex_x, hex_y, hex_z = hexagon[:,0], hexagon[:,1], hexagon[:,2]
                ax.plot3D(hex_x, hex_y, hex_z)
                rot_x, rot_y, rot_z = rotated_hexagon[:,0], rotated_hexagon[:,1], rotated_hexagon[:,2]
                ax.plot3D(rot_x, rot_y, rot_z)
                prot_x, prot_y, prot_z = param_rotated_hexagon[:,0], param_rotated_hexagon[:,1], param_rotated_hexagon[:,2]
                ax.plot3D(prot_x, prot_y, prot_z)
                ax.view_init(elev=90, azim=0)
                fig.show()
                plt.show()
print("Finished test")


"""fig = plt.figure()
ax= plt.axes(projection="3d")
hex_x, hex_y, hex_z = hexagon[:,0], hexagon[:,1], hexagon[:,2]
ax.plot3D(hex_x, hex_y, hex_z)
rot_x, rot_y, rot_z = rotated_hexagon[:,0], rotated_hexagon[:,1], rotated_hexagon[:,2]
ax.plot3D(rot_x, rot_y, rot_z)
ax.view_init(elev=90, azim=0)
fig.show()
plt.show()"""