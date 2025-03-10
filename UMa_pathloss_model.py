import numpy as np
import matplotlib.pyplot as plt
import random

fc = 3.5
hUT = 1.5
d_min, d_max = 10, 2000
d2D = np.linspace(d_min, d_max, 1000)
sigma_values = [5, 6, 8, 10, 12]
Pt_dbm = 12 + 30
P_threshold = -95
d_1km = 1000

def C_prime(h_UT):
    if h_UT <= 13:
        return 0
    elif 13 < h_UT <= 23:
        return ((h_UT - 13) / 10) ** 1.5
    else:
        raise ValueError("h_UT should be in the range 0 < h_UT ≤ 23 meters")

def Pr_LOS(d_2Dout, h_UT):
    if d_2Dout <= 18:
        return 1
    else:
        C_h_UT = C_prime(h_UT)
        term1 = (18 / d_2Dout) + np.exp(-d_2Dout / 63) * (1 - (18 / d_2Dout))
        term2 = (1 + C_h_UT * (5/4) * (d_2Dout / 100) ** 3 * np.exp(-d_2Dout / 150))
        return term1 * term2
    
def g(d_2D):
    if d_2D <= 18:
        return 0
    else:
        return (5/4) * (d_2D / 100) ** 3 * np.exp(-d_2D / 150)

def C(d_2D, h_UT):
    if h_UT < 13:
        return 0
    elif 13 <= h_UT <= 23:
        return ((h_UT - 13) / 10) ** 1.5 * g(d_2D)
    else:
        raise ValueError("h_UT must be in the range [1, 23]")

def h_E(d_2D, h_UT):
    c_value = C(d_2D, h_UT)
    prob = 1 / (1 + c_value)
    if random.random() < prob:
        return 1.0  # UMi or UMa with probability
    else:
        return random.choice(range(12, int(h_UT - 1.5) + 1,3))  # UMa case
    
def LOS_PL(fc, d2D, hUT):

    hBS = 25
    d3D = np.sqrt(d2D**2 + (hBS - hUT)**2)
    hE = h_E(d2D,hUT)
    hBS_prime = hBS - hE
    hUT_prime = hUT - hE
    c = 3e8
    d_BP_prime = 4 * hBS_prime * hUT_prime * fc * 1e9 / c

    PL = np.where(
        d2D <= d_BP_prime,
        28.0 + 22 * np.log10(d3D) + 20 * np.log10(fc),
        28.0 + 40 * np.log10(d3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP_prime**2 + (hBS - hUT)**2)
    )
    return PL

def LOS_PL_shadowing(fc, d2D, hUT):

    hBS = 25
    d3D = np.sqrt(d2D**2 + (hBS - hUT)**2)
    hE = h_E(d2D,hUT)
    hBS_prime = hBS - hE
    hUT_prime = hUT - hE
    c = 3e8
    d_BP_prime = 4 * hBS_prime * hUT_prime * fc * 1e9 / c

    PL = np.where(
        d2D <= d_BP_prime,
        28.0 + 22 * np.log10(d3D) + 20 * np.log10(fc),
        28.0 + 40 * np.log10(d3D) + 20 * np.log10(fc) - 9 * np.log10(d_BP_prime**2 + (hBS - hUT)**2)
    )
    return PL + np.random.normal(0,4)

def NLOS_PL(fc,d2D,hUT):
  hBS = 25
  d3D = np.sqrt(d2D**2 + (hBS-hUT)**2)
  PL = 32.4 + 20*np.log10(fc) + 30*np.log10(d3D)
  return PL

def NLOS_PL_shadowing(fc,d2D,hUT):
  hBS = 25
  d3D = np.sqrt(d2D**2 + (hBS-hUT)**2)
  PL = 32.4 + 20*np.log10(fc) + 30*np.log10(d3D)+ np.random.normal(0,7.8)
  return PL

def rx_power(PL,Ptx):
  return Ptx - PL
d_in = int(input("Enter the Distance from the Tx(m): "))
Ptx_in = int(input("Enter the Power of the Tx(dB): "))

p=Pr_LOS(d_in,hUT)

if random.random()<p:
  PL_in = LOS_PL_shadowing(fc,d_in,hUT)
else :
  PL_in = NLOS_PL_shadowing(fc,d_in,hUT)

print("The Received Power is(dB): ",rx_power(PL_in,Ptx_in))

PL1 = LOS_PL_shadowing(fc, d2D, hUT)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(d2D, PL1, label='Path Loss (dB)')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title('LOS Path Loss vs Distance (Linear Scale)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogx(d2D, PL1, label='Path Loss (dB)')
plt.xlabel('Distance (m, Log Scale)')
plt.ylabel('Path Loss (dB)')
plt.title('LOS Path Loss vs Distance (Log Scale)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

PL2 = NLOS_PL_shadowing(fc,d2D,hUT)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(d2D, PL2, label='Path Loss (dB)')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title('NLOS Path Loss vs Distance (Linear Scale)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(d2D, PL2, label='Path Loss (dB)')
plt.xlabel('Distance (m, Log Scale)')
plt.ylabel('Path Loss (dB)')
plt.title('NLOS Path Loss vs Distance (Log Scale)')
plt.grid(True)
plt.show()

def PL_model(fc, d2D, hUT):
    PL_values = []  # Store results for each element in d2D
    for d in d2D:
        p = Pr_LOS(d, hUT)  # Probability for this distance
        if random.random() < p:
            PL_values.append(LOS_PL(fc, d, hUT))  # LOS case
        else:
            PL_values.append(NLOS_PL(fc, d, hUT))  # NLOS case
    return np.array(PL_values)  # Convert to array if needed


PL3 = PL_model(fc, d2D, hUT)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(d2D, PL3, label='Path Loss (dB)')
plt.xlabel('Distance (m)')
plt.ylabel('Path Loss (dB)')
plt.title( 'Path Loss vs Distance (Linear Scale)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogx(d2D, PL3, label='Path Loss (dB)')
plt.xlabel('Distance (m, Log Scale)')
plt.ylabel('Path Loss (dB)')
plt.title('Path Loss vs Distance (Log Scale)')
plt.grid(True)
plt.show()

outage_probabilities1 = []
PL_1km = LOS_PL(fc, d_1km, hUT)

plt.figure(figsize=(8, 6))
for sigma_db in sigma_values:
    shadowing = np.random.normal(0, sigma_db, 1000)
    Pr_1km = Pt_dbm - PL_1km + shadowing
    P_out = np.mean(Pr_1km < P_threshold)
    outage_probabilities1.append(P_out)

    pdf, bins = np.histogram(Pr_1km, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, pdf, label=f'σ = {sigma_db} dB')



plt.xlabel('Received Power (dBm)')
plt.ylabel('Probability Density')
plt.title('PDF of Received Power at 1 km (LOS)')
plt.legend()
plt.grid(True)
plt.show()

print("\nOutage Probabilities Table:")
print("----------------------------------")
print("| Shadowing Sigma (dB) | P_out   |")
print("----------------------------------")
for sigma_db, P_out in zip(sigma_values, outage_probabilities1):
    print(f"| {sigma_db:<19} | {P_out:.3f} |")
print("----------------------------------")

plt.figure(figsize=(8, 6))
plt.plot(sigma_values, outage_probabilities1, marker='o', linestyle='-', color='b', label='Outage Probability')
plt.xlabel('Shadowing Standard Deviation (dB)')
plt.ylabel('Outage Probability')
plt.title('Outage Probability vs. Shadowing Standard Deviation (LOS)')
plt.grid(True)
plt.legend()
plt.show()

outage_probabilities2 = []
PL_1km = NLOS_PL(fc, d_1km, hUT)

plt.figure(figsize=(8, 6))
for sigma_db in sigma_values:
    shadowing = np.random.normal(0, sigma_db, 1000)
    Pr_1km = Pt_dbm - PL_1km + shadowing
    P_out = np.mean(Pr_1km < P_threshold)
    outage_probabilities2.append(P_out)

    pdf, bins = np.histogram(Pr_1km, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, pdf, label=f'σ = {sigma_db} dB')

plt.xlabel('Received Power (dBm)')
plt.ylabel('Probability Density')
plt.title('PDF of Received Power at 1 km (NLOS)')
plt.legend()
plt.grid(True)
plt.show()

print("\nOutage Probabilities Table:")
print("----------------------------------")
print("| Shadowing Sigma (dB) | P_out   |")
print("----------------------------------")
for sigma_db, P_out in zip(sigma_values, outage_probabilities2):
    print(f"| {sigma_db:<19} | {P_out:.3f} |")
print("----------------------------------")

plt.figure(figsize=(8, 6))
plt.plot(sigma_values, outage_probabilities2, marker='o', linestyle='-', color='b', label='Outage Probability')
plt.xlabel('Shadowing Standard Deviation (dB)')
plt.ylabel('Outage Probability')
plt.title('Outage Probability vs. Shadowing Standard Deviation (NLOS)')
plt.grid(True)
plt.legend()
plt.show()

outage_probabilities3 = []

def PL_model(fc, d2D, hUT):
    p = Pr_LOS(d2D, hUT)
    if random.random() < p:
      return LOS_PL(fc, d2D, hUT)  # LOS case
    else:
      return NLOS_PL(fc, d2D, hUT)  # NLOS case

plt.figure(figsize=(8, 6))
for sigma_db in sigma_values:
  Pr_1km = []
  for i in range(1000):
    PL4 = PL_model(fc,d_1km,hUT)
    shadowing = np.random.normal(0, sigma_db)
    Pr_1km.append(Pt_dbm - PL4 + shadowing)
  Pr_1km_new = np.array(Pr_1km)
  P_out = np.mean(Pr_1km_new < P_threshold)
  outage_probabilities3.append(P_out)

  pdf, bins = np.histogram(Pr_1km, bins=50, density=True)
  bin_centers = (bins[:-1] + bins[1:]) / 2
  plt.plot(bin_centers, pdf, label=f'σ = {sigma_db} dB')



plt.xlabel('Received Power (dBm)')
plt.ylabel('Probability Density')
plt.title('PDF of Received Power at 1 km')
plt.legend()
plt.grid(True)
plt.show()

print("\nOutage Probabilities Table:")
print("----------------------------------")
print("| Shadowing Sigma (dB) | P_out   |")
print("----------------------------------")
for sigma_db, P_out in zip(sigma_values, outage_probabilities3):
    print(f"| {sigma_db:<19} | {P_out:.3f} |")
print("----------------------------------")

plt.figure(figsize=(8, 6))
plt.plot(sigma_values, outage_probabilities3, marker='o', linestyle='-', color='b', label='Outage Probability')
plt.xlabel('Shadowing Standard Deviation (dB)')
plt.ylabel('Outage Probability')
plt.title('Outage Probability vs. Shadowing Standard Deviation)')
plt.grid(True)
plt.legend()
plt.show()