import numpy as np
import matplotlib.pyplot as plt

games = 10000 # number of games simulated
doors = np.array(['P', 'N', 'N']) # P = Prize N = Nothing
setup_doors = np.array([np.random.permutation(doors) for _ in range(games)]) # Randomly arrange the doors

# Initial choice of the player (before any door is opened)
# Strategy A: Stays with this choice, Strategy B: Switches to the other door
initial_choice = np.random.randint(0, 3, size=games)
indices = np.arange(3)

# Host opens a door that has 'N' and is NOT the player's initial choice
mask_host = (setup_doors == 'N') & (indices != initial_choice[:, None])
door_opened = [np.random.choice(np.where(ii)[0]) for ii in mask_host]
setup_doors[np.arange(games), door_opened] = 'A' # Mark the opened door as 'A'

# Strategy B: Switch door (choose the door that is neither the initial choice nor opened)
mask_switch = (setup_doors != 'A') & (indices != initial_choice[:, None])
choice_B = np.array([np.random.choice(np.where(ii)[0]) for ii in mask_switch])
choice_A = initial_choice

# Determine wins (True if the chosen door contains 'P')
win_A = setup_doors[np.arange(games), choice_A] == 'P'
win_B = setup_doors[np.arange(games), choice_B] == 'P'

csi_A = 1*win_A
csi_B = 1*win_B

ii = np.arange(1, games + 1)
samp_av_A = np.cumsum(csi_A)/ii
print(f'Final Win Rate (Stay): {samp_av_A[-1]:.3f}')
samp_av_B = np.cumsum(csi_B)/ii
print(f'Final Win Rate (Switch): {samp_av_B[-1]:.3f}')
samp_av2_A = np.cumsum(csi_A**2) / ii
vari_A = (samp_av2_A - samp_av_A**2) * ii/(ii-1)
vari_A[0] = 0
rsd_A = np.sqrt(vari_A/ii)/abs(samp_av_A)
PRSD_A = rsd_A*100
print(f"Final PRSD (Stay): {PRSD_A[-1]:.3f} %")
samp_av2_B = np.cumsum(csi_B**2) / ii
vari_B = (samp_av2_B - samp_av_B**2) * ii/(ii-1)
vari_B[0] = 0
rsd_B = np.sqrt(vari_B/ii)/abs(samp_av_B)
PRSD_B = rsd_B*100
print(f"Final PRSD (Switch): {PRSD_B[-1]:.3f} %")

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.plot(samp_av_A, label = 'Strategy: Stay (1/3)')
plt.plot(samp_av_B, label = 'Strategy: Switch (2/3)')
plt.ylabel('Win Rate (Sample Average)')
plt.xlabel('Number of Games')
plt.title('Monty Hall Simulation: Convergence of Win Rates')
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(PRSD_A, label= 'PRSD Stay')
plt.plot(PRSD_B, label= 'PRSD Switch')
plt.axhline(y=1.0, color='r', linestyle='--', label='1% Threshold')
plt.ylabel('PRSD %')
plt.xlabel('Number of Games')
plt.title('Precision: Percent Relative Standard Deviation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()