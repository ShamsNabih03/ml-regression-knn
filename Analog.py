import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- Parameters ---
Fs = 2_000_000   # Sampling rate (Hz)
fc = 100_000     # Carrier frequency (Hz)
fm = 1_000       # Message frequency (Hz)
Am, Ac, mu = 1.0, 1.0, 0.6
t = np.arange(0, 0.006, 1/Fs)

# --- Signal Generation ---
m = Am * np.cos(2*np.pi*fm*t)                     # Message
carrier = Ac * np.cos(2*np.pi*fc*t)
s = Ac * (1 + mu * m/Am) * np.cos(2*np.pi*fc*t)  # AM signal

# --- Envelope Detector ---
def rc_lowpass(x, R, C, Fs):
    alpha = 1 / (1 + 1/(2*np.pi*R*C*Fs))
    y = np.zeros_like(x)
    for n in range(1, len(x)):
        y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    return y

tau = 1e-4     # From 1/fc << tau << 1/fm
C_env = 100e-9
R_env = tau / C_env
env = rc_lowpass(np.abs(s), R_env, C_env, Fs)
env -= np.mean(env)

# --- Coherent Demodulation ---
C_lpf, f_cut = 10e-9, 1.5*fm
R_lpf = 1 / (2*np.pi*C_lpf*f_cut)
product = s * 2*np.cos(2*np.pi*fc*t)
lpf = rc_lowpass(product, R_lpf, C_lpf, Fs)
lpf -= np.mean(lpf)

# --- Plots ---
plt.figure(figsize=(8,3))
plt.plot(t[:2000], s[:2000])
plt.title("DSB-TC AM Signal (Time Domain)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.show()

plt.figure(figsize=(8,3))
plt.plot(t[:2000], env[:2000])
plt.title("Envelope Detector Output"); plt.xlabel("Time (s)"); plt.show()

plt.figure(figsize=(8,3))
plt.plot(t[:2000], lpf[:2000])
plt.title("Coherent Detector Output"); plt.xlabel("Time (s)"); plt.show()

# --- Display Component Values ---
print(f"Envelope Detector: R = {R_env:.0f} Ω, C = {C_env:.2e} F, τ = {tau:.2e} s")
print(f"Coherent LPF: R = {R_lpf:.0f} Ω, C = {C_lpf:.2e} F, f_c = {f_cut:.1f} Hz")
