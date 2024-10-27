import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import lti, step, lsim
from sympy import symbols, Function, Eq, dsolve, laplace_transform, Heaviside, sin
from sympy.abc import t, s
from sympy import Rational, simplify
from scipy.signal import StateSpace, step

# example form me
m = 1.0  # Mass for spring-mass system
k = 20.0  # Spring constant for spring-mass system
L = 1.0  # Inductance for RLC circuit
C = 0.1  # Capacitance for RLC circuit

b_critical = 2 * np.sqrt(m * k)  # Critical damping (ξ = 1)
b_over = b_critical * 1.5  # Over damped (ξ > 1)
b_under = b_critical * 0.5  # Under damped (ξ < 1)
b_values = {'Overdamped': b_over, 'Critically Damped': b_critical, 'Underdamped': b_under}

R_critical = 2 * np.sqrt(L / C)  # Critical damping for RLC circuit
R_over = R_critical * 1.5  # Over damped for RLC
R_under = R_critical * 0.5  # Under damped for RLC
R_values = {'Overdamped': R_over, 'Critically Damped': R_critical, 'Underdamped': R_under}

# Time range for simulation & Time evaluation points
t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)


# Define each input type
def unit_step(t):
    return 1

def unit_ramp(t):
    return t

def sinusoidal(t, omega=1.0):
    return np.sin(omega * t)

# Spring-Mass System ODE definition with force function
def spring_mass_system(t, y, m, b, k, F_func):
    x, v = y  # x is displacement, v is velocity
    F = F_func(t)  # Evaluate the force function at time t
    dxdt = v
    dvdt = (F - b * v - k * x) / m
    return [dxdt, dvdt]

# RLC Circuit ODE definition with voltage function
def rlc_circuit_system(t, y, L, R, C, V_func):
    I, dI_dt = y  # I is current, dI_dt is rate of change of current
    V_prime = V_func(t)  # Evaluate the voltage function at time t
    d2I_dt2 = (V_prime - R * dI_dt - (1 / C) * I) / L
    return [dI_dt, d2I_dt2]


# Simulate Spring-Mass System for Each Damping Condition and Input Type
for damping_type, b in b_values.items():
    print(f"--- Spring-Mass System: {damping_type} Damping ---")

    for F_func, label in [(unit_step, 'Unit Step'), (unit_ramp, 'Unit Ramp'),
                          (lambda t: sinusoidal(t, omega=1.0), 'Sinusoidal (ω=1.0)')]:
        solution = solve_ivp(spring_mass_system, t_span, [0.0, 0.0], args=(m, b, k, F_func), t_eval=t_eval)

        plt.plot(solution.t, solution.y[0], label=f"{label} Input")
    plt.xlabel('Time')
    plt.ylabel('Displacement (x)')
    plt.title(f"{damping_type} Damping - Spring-Mass System - Hamza 1211162")
    plt.legend()
    plt.show()

# Simulate RLC Circuit for Each Damping Condition and Input Type
for damping_type, R in R_values.items():
    print(f"--- RLC Circuit: {damping_type} Damping ---")

    for V_func, label in [(unit_step, 'Unit Step'), (unit_ramp, 'Unit Ramp'),
                          (lambda t: sinusoidal(t, omega=1.0), 'Sinusoidal (ω=1.0)')]:
        solution_rlc = solve_ivp(rlc_circuit_system, t_span, [0.0, 0.0], args=(L, R, C, V_func), t_eval=t_eval)

        plt.plot(solution_rlc.t, solution_rlc.y[0], label=f"{label} Input")
    plt.xlabel('Time')
    plt.ylabel('Current (I)')
    plt.title(f"{damping_type} Damping - RLC Circuit - Hamza 1211162")
    plt.legend()
    plt.show()

# Laplace Domain Transfer Functions (Symbolic)
s = symbols('s')
X, F = symbols('X F')
omega_n = np.sqrt(k / m)  # Natural frequency for spring-mass

# Transfer Function for Spring-Mass System
H_s_spring_mass = 1 / (m * s**2 + b_critical * s + k)
simplified_H_s_spring_mass = simplify(H_s_spring_mass)
print(f"Spring-Mass System Transfer Function: {simplified_H_s_spring_mass} - Hamza 1211162")

# Transfer Function for RLC Circuit
H_s_rlc = 1 / (L * s**2 + R_critical * s + (1 / C))
simplified_H_s_rlc = simplify(H_s_rlc)
print(f"RLC Circuit Transfer Function: {simplified_H_s_rlc} - Hamza 1211162")

# Spring-Mass System State-Space
for damping_type, b in b_values.items():
    A_spring_mass = np.array([[0, 1], [-k / m, -b / m]])
    B_spring_mass = np.array([[0], [1 / m]])
    C_spring_mass = np.array([[1, 0]])
    D_spring_mass = np.array([[0]])

    system_spring_mass = StateSpace(A_spring_mass, B_spring_mass, C_spring_mass, D_spring_mass)

    t, y = step(system_spring_mass, T=t_eval)
    plt.plot(t, y, label=f"{damping_type} Damping - Step Response - Hamza 1211162")

plt.xlabel('Time')
plt.ylabel('Displacement (x)')
plt.title("State-Space Responses for Spring-Mass System - Hamza 1211162")
plt.legend()
plt.show()

# RLC Circuit State-Space
for damping_type, R in R_values.items():
    A_rlc = np.array([[0, 1], [-1 / (L * C), -R / L]])
    B_rlc = np.array([[0], [1 / L]])
    C_rlc = np.array([[1, 0]])
    D_rlc = np.array([[0]])

    system_rlc = StateSpace(A_rlc, B_rlc, C_rlc, D_rlc)
    t, y = step(system_rlc, T=t_eval)
    plt.plot(t, y, label=f"{damping_type} Damping - Step Response")

plt.xlabel('Time')
plt.ylabel('Current (I)')
plt.title("State-Space Responses for RLC Circuit - Hamza 1211162")
plt.legend()
plt.show()