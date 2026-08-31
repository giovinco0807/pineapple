import subprocess

def run_solver():
    # Board: {'top': ['2h', '2s'], 'middle': ['3d', '3c', '9c', '3s'], 'bottom': ['Td', 'Tc', 'Ts']}
    # Dealt: ['3h', 'Jh', 'X2']
    # Create an input string for the solver (t0 format or stdin format)
    # Actually, we can use the backward stdin protocol to ask for the EV of specific moves.
    pass

if __name__ == "__main__":
    run_solver()
