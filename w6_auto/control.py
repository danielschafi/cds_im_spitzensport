from cyipopt import minimize_ipopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize



# CONSTANTS


# Fahrzeug
m = 790 #kg
F_MotorMax = 10.41 #kN
alpha = 0.001872 # 1/m

def F_HaftMax(age:str="new"):
    if age=="old":
        return 9.88 #kN
    elif age=="new":
        return 12.27
tire_age = "new"

# Config Constants
L = 20
B = 1
dL = 1
xStart = 0.1
vStart = 0.2
N=int(L/dL)
num_vars_t =2
optimization_start_t = 1

u = np.zeros((N,num_vars_t)) # 0=x, 1=v
u[0] = [xStart, vStart]





# Physics Functions
def s(u: np.ndarray, t:int):
    """strecke"""
    x = u[:,0]
    return np.sqrt(dL**2 + (x[t-1]-x[t])**2)

def vm(u:np.ndarray, t:int):
    """durchschnittsgeschwindigkeit"""
    v = u[:,1]
    return (v[t-1] + v[t]) / 2

def vx(u:np.ndarray, t:int):
    x = u[:,0]
    return (x[t]-x[t-1])/dt(u, t)


def vy(u:np.ndarray, t:int):
    return dL/dt(u, t)

def ax(u:np.ndarray, t:int):
    return(vx(u,t), vx(u,t-1))/dt(u,t-1) # why t-1???

def ay(u:np.ndarray, t:int):
    return(vy(u,t), vy(u,t-1))/dt(u,t-1) # why t-1???

def a(u:np.ndarray, t:int):
    return np.sqrt(ax(u,t)**2 + ay(u,t)**2)

def dt(u:np.ndarray, t:int):
    """Gebrauchte Zeit für den abschnitt"""
    return s(u, t) / vm(u, t)

def a(u:np.ndarray, t:int):
    """Beschleunigung """
    return (u[t,1]- u[t-1,1]) / dt(u,t)    

def F_Brems_Motor(u:np.ndarray, t:int):
    return m * (a(u, t)+ alpha* vm(u, t)**2)

def F_Reifen(u:np.ndarray, t:int):
    return m* a(u, t)



# Define Optimization Problem

def objective(u:np.ndarray):
    """
    The Objective function T* = min [ dt(1) + dt(2) +... ]
    minimize time
    """
    
    return np.sum([dt(np.concatenate([np.array([[xStart,vStart]]), u.reshape((N-1, num_vars_t))], axis=0), t) for t in range(1, L)])

def box_constraints() -> Bounds:
    """Bounds on input variables x,v"""
    # 0 <= x(i) <= B
    # 0.001 <= v(i) <= infty
    # for each t
    return Bounds(
        lb=[0.0 for _ in range(optimization_start_t, N)] + [0.001 for _ in range(optimization_start_t, N)],
        ub=[float(B) for _ in range(optimization_start_t, N)] + [np.inf for _ in range(optimization_start_t, N)]
    )


# Non linear Constraints
# To get functions to optimize at each t
# And receive u as passed by ipopt (flat) -> we need to reshape
def F_Brems_Motor_t(t:int):
    """returns the F_Brems_Motor Constraint function for time t, takes the flattened u as input"""
    return lambda u: F_Brems_Motor(np.concatenate([np.array([[xStart,vStart]]), u.reshape((N-1, num_vars_t))], axis=0), t)

def F_Reifen_t(t:int):
    """returns the F_Reifen Constraint function for time t, takes the flattened u as input"""
    return lambda u: F_Reifen(np.concatenate([np.array([[xStart,vStart]]), u.reshape((N-1, num_vars_t))], axis=0), t)


def nonlinear_constraints() -> list[NonlinearConstraint]:
    """
    Returns the nonlinear constraints
    for each timestep t the Bems motor, Reifen constraints need to be fulfilled
    Theird order does not matter, the optimizer will optimize all variables at the same time anyways and not treat this as a sequential problem
    """
    constr = []
    for t in range(optimization_start_t, N):
        constr.append(
            NonlinearConstraint(
            fun=F_Brems_Motor_t(t),
            lb=-np.inf,
            ub=F_MotorMax)
        )
        constr.append(
            NonlinearConstraint(
                fun=F_Reifen_t(t),
                lb=0,
                ub=F_HaftMax(tire_age)
            ) # in den slides ist function und ub **2 aber das ist ja das selbe
        )
    return constr



def objects_on_track():
    obj = []
    
    x_dims = [0.0, 0.5]
    y_dims = [5.0, 8]
    
    r1 = Rectangle(xy=[np.mean(x_dims),np.mean(y_dims)],
              width=0.6,
              height=4)
    
    
    pass
def linear_constraints() -> list[LinearConstraint]:
    """
    Returns the linear constraints
    """
    object_start_t = 5
    object_end_t = 10
    object_start_x=0.0
    object_end_x=0.5
    
    constr = []
    for t in range(optimization_start_t, N):
        if t >= object_start_t and t <= object_end_t:
            constr.append(
                LinearConstraint( # u
                    A=np.array([[1, 0]]),
                    lb=np.array([object_start_x]),
                    ub=np.array([object_end_x])
                )
            )
        
    return constr




def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def plot_optim_path(result:np.ndarray, objects: list ):
    
    fig, ax = plt.subplots()

    ax.add_patch(Rectangle((0, 0), L, B, edgecolor='black', facecolor='lightgrey'))

    for obj in objects:
        ax.add_patch(Rectangle((obj[0], obj[1]), obj[2], obj[3], edgecolor='black', facecolor='blue'))

    plt.xlim(0, L)
    plt.ylim(0, B)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Optimierungspfad")
    plt.grid()
    plt.show()


def solve_scipty():
    pass

def main():
    nonlin_constraints = nonlinear_constraints()
    box_constr = box_constraints()


    result = minimize_ipopt(
        fun=objective, 
        x0=u[optimization_start_t:].flatten(), 
        bounds=box_constr, 
        constraints=nonlin_constraints
    )

    print(result)
    # cant we just use scipy?
    optimized_result = result.x
    optimized_result_reshaped = np.concatenate([np.array([[xStart,vStart]]), optimized_result.reshape((N-1, num_vars_t))], axis=0)
    print("Optimized result (reshaped):", optimized_result_reshaped)

    x = optimized_result_reshaped[:,0]
    v = optimized_result_reshaped[:,1]
    y = np.linspace(0, L, N)
    
    # plt.title(f"Entwicklung von x und v über die zeit t, mit {N} Abschnitten")
    # plt.plot(list(range(len(x))), x, label="x")
    # plt.plot(list(range(len(v))), v, label="v")
    # # plt.plot(optimized_result_reshaped)
    # plt.legend()
    # plt.xlabel("t")
    # plt.ylabel("x/v")
    # plt.show()
    
    fig1, ax1 = plt.subplots()
    lines = colored_line(x, y, v, ax1, linewidth=10, cmap="plasma")
    fig1.colorbar(lines)  # add a color legend

    # Set the axis limits and tick positions
    ax1.set_xlim(0, B)
    ax1.set_ylim(0, L)
    ax1.set_xticks((0, B))
    ax1.set_yticks((0, L))
    ax1.set_title("Color at each point")

    plt.show()



if __name__ == "__main__":
    main()


