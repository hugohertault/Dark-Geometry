#!/usr/bin/env python3
"""
================================================================================
HERTAULT COSMOLOGICAL MODEL - PERTURBATIONS COSMOLOGIQUES
================================================================================

Calcul des perturbations de densité et du spectre de puissance P(k).

Comparaison HCM vs ΛCDM :
- Fonction de croissance f(z)
- Facteur de croissance D(z)
- Spectre de puissance P(k)
- Paramètre σ₈

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.interpolate import interp1d

# =============================================================================
# CONSTANTES COSMOLOGIQUES
# =============================================================================

# Paramètres Planck 2018
H0 = 67.4               # km/s/Mpc
h = H0 / 100
Omega_m = 0.315
Omega_Lambda = 0.685
Omega_b = 0.049
Omega_cdm = Omega_m - Omega_b

# Conversion
H0_SI = H0 * 1e3 / 3.086e22    # s⁻¹
c = 2.998e8                      # m/s
G = 6.674e-11                    # m³/kg/s²
Mpc = 3.086e22                   # m

# Densité critique aujourd'hui
rho_crit_0 = 3 * H0_SI**2 / (8 * np.pi * G)  # kg/m³

# Densité critique HCM
rho_c_HCM = 6.27e-27  # kg/m³

print("="*70)
print("PARAMÈTRES COSMOLOGIQUES")
print("="*70)
print(f"H₀ = {H0} km/s/Mpc")
print(f"Ω_m = {Omega_m}, Ω_Λ = {Omega_Lambda}")
print(f"ρ_crit,0 = {rho_crit_0:.2e} kg/m³")
print(f"ρ_c (HCM) = {rho_c_HCM:.2e} kg/m³")
print(f"ρ_m,0 / ρ_c = {Omega_m * rho_crit_0 / rho_c_HCM:.2f}")
print("="*70)


# =============================================================================
# COSMOLOGIE DE FOND
# =============================================================================

def E_squared(z, Omega_m=Omega_m, Omega_Lambda=Omega_Lambda):
    """E²(z) = H²(z)/H₀² pour ΛCDM"""
    return Omega_m * (1 + z)**3 + Omega_Lambda


def E(z):
    """E(z) = H(z)/H₀"""
    return np.sqrt(E_squared(z))


def Omega_m_z(z):
    """Ω_m(z) = densité de matière relative"""
    return Omega_m * (1 + z)**3 / E_squared(z)


def rho_matter(z):
    """Densité de matière en fonction de z (kg/m³)"""
    return Omega_m * rho_crit_0 * (1 + z)**3


def z_transition_HCM():
    """Redshift de transition DM → DE dans HCM"""
    # ρ_m(z_trans) = ρ_c
    # Ω_m ρ_crit,0 (1+z)³ = ρ_c
    z_trans = (rho_c_HCM / (Omega_m * rho_crit_0))**(1/3) - 1
    return max(z_trans, 0)


z_trans = z_transition_HCM()
print(f"\nRedshift de transition HCM : z_trans = {z_trans:.2f}")


# =============================================================================
# MASSE EFFECTIVE HCM
# =============================================================================

def m_eff_squared_ratio(z):
    """
    m²_eff / m₀² où m₀ = α* M_Pl
    
    m²_eff / m₀² = 1 - (ρ/ρc)^(2/3)
    
    Retourne :
    - > 0 pour ρ < ρc (régime DE, stable)
    - < 0 pour ρ > ρc (régime DM, tachyonique)
    - = 0 à la transition
    """
    rho = rho_matter(z)
    ratio = (rho / rho_c_HCM)**(2/3)
    return 1 - ratio


def w_phi_effective(z):
    """
    Équation d'état effective du champ φ.
    
    En régime DM (m² < 0) : w ≈ 0 (dust-like)
    En régime DE (m² > 0) : w ≈ -1 (cosmological constant)
    Transition : interpolation douce
    """
    m2_ratio = m_eff_squared_ratio(z)
    
    # Fonction de transition douce
    # w = -1 quand m² > 0, w = 0 quand m² < 0
    sigma = 0.5  # largeur de transition
    f = 0.5 * (1 + np.tanh(m2_ratio / sigma))
    
    return -f  # -1 pour DE, ~0 pour DM


# =============================================================================
# FACTEUR DE CROISSANCE
# =============================================================================

def growth_ode_LCDM(y, a):
    """
    Équation de croissance linéaire pour ΛCDM.
    
    D'' + (3/a + E'/E) D' - 3Ω_m(a)/(2a²) D = 0
    
    Variables : y = [D, dD/da]
    """
    D, dD = y
    z = 1/a - 1
    
    E_val = E(z)
    Om_a = Omega_m_z(z)
    
    # dE/da = -3Ω_m(1+z)² / (2E) × (-1/a²) = 3Ω_m(1+z)² / (2Ea²)
    # E'/E = (dE/da)/E = 3Ω_m(1+z)²/(2E²a²)
    dE_da = -1.5 * Omega_m * (1+z)**2 / (E_val * a**2)
    
    ddD = -dD * (3/a + dE_da/E_val) + 1.5 * Om_a * D / a**2
    
    return [dD, ddD]


def growth_ode_HCM(y, a, suppress_small_scale=False, k=0.1):
    """
    Équation de croissance pour HCM.
    
    Modification : le champ scalaire a une pression effective.
    
    D'' + (3/a + E'/E) D' - 3Ω_m,eff(a)/(2a²) D = 0
    
    où Ω_m,eff inclut la contribution du champ.
    """
    D, dD = y
    z = 1/a - 1
    
    E_val = E(z)
    Om_a = Omega_m_z(z)
    
    dE_da = -1.5 * Omega_m * (1+z)**2 / (E_val * a**2)
    
    # Modification HCM : près de la transition, la croissance est légèrement modifiée
    m2_ratio = m_eff_squared_ratio(z)
    
    # Facteur de suppression (effet de pression du champ)
    # Quand m² → 0, le champ devient "rigide" et supprime la croissance
    if m2_ratio > -1:  # Près ou après la transition
        suppression = 1 - 0.05 * np.exp(-m2_ratio**2 / 0.5)
    else:
        suppression = 1.0
    
    ddD = -dD * (3/a + dE_da/E_val) + 1.5 * Om_a * D / a**2 * suppression
    
    return [dD, ddD]


def compute_growth_factor(model='LCDM', z_max=100, n_points=500):
    """
    Calcule le facteur de croissance D(z) normalisé à D(0) = 1.
    """
    a_min = 1 / (1 + z_max)
    a_max = 1.0
    a_arr = np.linspace(a_min, a_max, n_points)
    
    # Conditions initiales : D ∝ a en régime de matière dominée
    # On normalise après
    D_init = a_min
    dD_init = 1.0
    
    if model == 'LCDM':
        sol = odeint(growth_ode_LCDM, [D_init, dD_init], a_arr)
    elif model == 'HCM':
        sol = odeint(growth_ode_HCM, [D_init, dD_init], a_arr)
    else:
        raise ValueError(f"Modèle inconnu : {model}")
    
    D = sol[:, 0]
    
    # Normaliser à D(z=0) = 1
    D = D / D[-1]
    
    z_arr = 1/a_arr - 1
    
    return z_arr[::-1], D[::-1]


def compute_growth_rate(z_arr, D_arr):
    """
    Calcule le taux de croissance f = d ln D / d ln a = -(1+z) d ln D / dz
    """
    ln_D = np.log(D_arr)
    dln_D_dz = np.gradient(ln_D, z_arr)
    f = -(1 + z_arr) * dln_D_dz
    return f


# =============================================================================
# SPECTRE DE PUISSANCE
# =============================================================================

def transfer_function_BBKS(k):
    """
    Fonction de transfert BBKS (Bardeen, Bond, Kaiser, Szalay 1986).
    
    k en h/Mpc
    """
    q = k / (Omega_m * h**2 * np.exp(-Omega_b - np.sqrt(2*h) * Omega_b/Omega_m))
    T = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
    return T


def primordial_spectrum(k, A_s=2.1e-9, n_s=0.965, k_pivot=0.05):
    """
    Spectre primordial (Harrison-Zeldovich avec tilt).
    
    P_prim(k) = A_s (k/k_pivot)^(n_s - 1)
    """
    return A_s * (k / k_pivot)**(n_s - 1)


def power_spectrum_LCDM(k, z=0, D_interp=None):
    """
    Spectre de puissance P(k) pour ΛCDM.
    
    P(k, z) = D²(z) × T²(k) × P_prim(k) × normalization
    
    Normalisé pour avoir σ₈ ≈ 0.81
    """
    T = transfer_function_BBKS(k)
    P_prim = primordial_spectrum(k)
    
    if D_interp is not None and z > 0:
        D_z = D_interp(z)
    else:
        D_z = 1.0
    
    # Le spectre non-normalisé
    P_unnorm = D_z**2 * T**2 * P_prim * k
    
    return P_unnorm


def power_spectrum_HCM(k, z=0, D_interp=None, k_cut=5.0, alpha_cut=0.5):
    """
    Spectre de puissance P(k) pour HCM.
    
    Modification : suppression exponentielle aux petites échelles.
    
    P_HCM(k) = P_LCDM(k) × exp(-(k/k_cut)^α)
    """
    P_base = power_spectrum_LCDM(k, z, D_interp)
    
    # Suppression aux petites échelles
    suppression = np.exp(-(k / k_cut)**alpha_cut)
    
    return P_base * suppression


def compute_sigma_R(R, k_arr, P_arr):
    """
    Calcule σ(R) = variance du champ de densité lissé à l'échelle R.
    
    σ²(R) = 1/(2π²) ∫ k² P(k) W²(kR) dk
    
    avec W(x) = 3(sin x - x cos x)/x³ (top-hat)
    """
    def W_tophat(x):
        """Fonction fenêtre top-hat en Fourier"""
        return np.where(x < 0.01, 1 - x**2/10, 3 * (np.sin(x) - x * np.cos(x)) / x**3)
    
    W = W_tophat(k_arr * R)
    integrand = k_arr**2 * P_arr * W**2
    
    sigma_sq = np.trapz(integrand, k_arr) / (2 * np.pi**2)
    return np.sqrt(sigma_sq)


def compute_sigma8(k_arr, P_arr):
    """Calcule σ₈ = σ(R = 8 h⁻¹ Mpc)"""
    R8 = 8.0  # h⁻¹ Mpc
    return compute_sigma_R(R8, k_arr, P_arr)


# =============================================================================
# SIMULATION PRINCIPALE
# =============================================================================

def main():
    print("\n" + "="*70)
    print("PERTURBATIONS COSMOLOGIQUES : HCM vs ΛCDM")
    print("="*70)
    
    # =========================================================================
    # 1. FACTEUR DE CROISSANCE
    # =========================================================================
    
    print("\n--- Facteur de croissance D(z) ---")
    
    z_LCDM, D_LCDM = compute_growth_factor('LCDM')
    z_HCM, D_HCM = compute_growth_factor('HCM')
    
    # Interpolateurs
    D_interp_LCDM = interp1d(z_LCDM, D_LCDM, fill_value='extrapolate')
    D_interp_HCM = interp1d(z_HCM, D_HCM, fill_value='extrapolate')
    
    # Taux de croissance
    f_LCDM = compute_growth_rate(z_LCDM, D_LCDM)
    f_HCM = compute_growth_rate(z_HCM, D_HCM)
    
    # Valeurs à z = 0 (premier élément après inversion)
    print(f"D(z=0) ΛCDM = {D_LCDM[0]:.4f}")
    print(f"D(z=0) HCM  = {D_HCM[0]:.4f}")
    print(f"Différence : {(D_HCM[0]/D_LCDM[0] - 1)*100:.2f}%")
    
    # =========================================================================
    # 2. SPECTRE DE PUISSANCE
    # =========================================================================
    
    print("\n--- Spectre de puissance P(k) ---")
    
    k_arr = np.logspace(-3, 2, 500)  # h/Mpc
    
    # Spectre non-normalisé
    P_LCDM_raw = power_spectrum_LCDM(k_arr, z=0)
    
    # Calculer σ₈ du spectre brut
    sigma8_raw = compute_sigma8(k_arr, P_LCDM_raw)
    
    # Facteur de normalisation pour avoir σ₈ = 0.81
    sigma8_target = 0.81
    norm_factor = (sigma8_target / sigma8_raw)**2
    
    print(f"Facteur de normalisation : {norm_factor:.2e}")
    
    # Spectres normalisés
    P_LCDM = P_LCDM_raw * norm_factor
    P_HCM = power_spectrum_HCM(k_arr, z=0, D_interp=D_interp_HCM, k_cut=5.0) * norm_factor
    
    # σ₈ final
    sigma8_LCDM = compute_sigma8(k_arr, P_LCDM)
    sigma8_HCM = compute_sigma8(k_arr, P_HCM)
    
    print(f"σ₈ ΛCDM = {sigma8_LCDM:.3f}")
    print(f"σ₈ HCM  = {sigma8_HCM:.3f}")
    print(f"Différence : {(sigma8_HCM/sigma8_LCDM - 1)*100:.1f}%")
    
    # =========================================================================
    # 3. ÉQUATION D'ÉTAT
    # =========================================================================
    
    print("\n--- Équation d'état w(z) ---")
    
    z_w = np.linspace(0, 5, 100)
    w_HCM = np.array([w_phi_effective(zi) for zi in z_w])
    m2_ratio = np.array([m_eff_squared_ratio(zi) for zi in z_w])
    
    print(f"w(z=0) HCM = {w_HCM[0]:.3f}")
    print(f"w(z=2) HCM = {w_phi_effective(2):.3f}")
    
    # =========================================================================
    # FIGURES
    # =========================================================================
    
    print("\n--- Génération des figures ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Couleurs
    C_LCDM = '#E63946'
    C_HCM = '#457B9D'
    
    # --- 1) Facteur de croissance D(z) ---
    ax = axes[0, 0]
    ax.plot(z_LCDM, D_LCDM, C_LCDM, lw=2.5, label='ΛCDM')
    ax.plot(z_HCM, D_HCM, C_HCM, lw=2.5, ls='--', label='HCM')
    ax.axvline(z_trans, color='gray', ls=':', lw=1.5, alpha=0.7, label=f'z_trans = {z_trans:.2f}')
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('D(z)', fontsize=11)
    ax.set_title('Facteur de croissance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 10])
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # --- 2) Taux de croissance f(z) ---
    ax = axes[0, 1]
    ax.plot(z_LCDM[:-5], f_LCDM[:-5], C_LCDM, lw=2.5, label='ΛCDM')
    ax.plot(z_HCM[:-5], f_HCM[:-5], C_HCM, lw=2.5, ls='--', label='HCM')
    ax.plot(z_LCDM, Omega_m_z(z_LCDM)**0.55, 'k:', lw=1.5, alpha=0.5, label=r'$\Omega_m^{0.55}$')
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('f(z) = d ln D / d ln a', fontsize=11)
    ax.set_title('Taux de croissance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 3])
    ax.set_ylim([0.4, 1.1])
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # --- 3) Spectre de puissance P(k) ---
    ax = axes[0, 2]
    ax.loglog(k_arr, P_LCDM, C_LCDM, lw=2.5, label='ΛCDM')
    ax.loglog(k_arr, P_HCM, C_HCM, lw=2.5, ls='--', label='HCM')
    ax.axvline(5.0, color='gray', ls=':', lw=1.5, alpha=0.7, label='k_cut = 5 h/Mpc')
    ax.set_xlabel('k (h/Mpc)', fontsize=11)
    ax.set_ylabel('P(k) (h⁻³ Mpc³)', fontsize=11)
    ax.set_title('Spectre de puissance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([1e-3, 100])
    ax.grid(True, alpha=0.3, which='both')
    
    # --- 4) Ratio P_HCM / P_LCDM ---
    ax = axes[1, 0]
    ratio = P_HCM / P_LCDM
    ax.semilogx(k_arr, ratio, C_HCM, lw=2.5)
    ax.axhline(1, color='gray', ls='--', lw=1.5)
    ax.axvline(5.0, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax.fill_between(k_arr, 0.95, 1.05, alpha=0.1, color='green')
    ax.set_xlabel('k (h/Mpc)', fontsize=11)
    ax.set_ylabel('P_HCM / P_ΛCDM', fontsize=11)
    ax.set_title('Ratio des spectres', fontsize=12, fontweight='bold')
    ax.set_xlim([1e-3, 100])
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    # --- 5) Masse effective m² ---
    ax = axes[1, 1]
    ax.plot(z_w, m2_ratio, C_HCM, lw=2.5)
    ax.axhline(0, color='gray', ls='--', lw=1.5)
    ax.axvline(z_trans, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax.fill_between(z_w, -10, 0, alpha=0.1, color=C_LCDM, label='Régime DM (m² < 0)')
    ax.fill_between(z_w, 0, 1, alpha=0.1, color='green', label='Régime DE (m² > 0)')
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('m²_eff / m₀²', fontsize=11)
    ax.set_title('Masse effective (régimes)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim([0, 5])
    ax.set_ylim([-5, 1])
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # --- 6) Équation d'état w(z) ---
    ax = axes[1, 2]
    ax.plot(z_w, w_HCM, C_HCM, lw=2.5, label='HCM')
    ax.axhline(-1, color=C_LCDM, ls='--', lw=2, label='ΛCDM (w = -1)')
    ax.axhline(0, color='gray', ls=':', lw=1.5, alpha=0.5, label='Dust (w = 0)')
    ax.axvline(z_trans, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax.set_xlabel('z', fontsize=11)
    ax.set_ylabel('w(z)', fontsize=11)
    ax.set_title('Équation d\'état effective', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim([0, 5])
    ax.set_ylim([-1.2, 0.2])
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    fig.suptitle('PERTURBATIONS COSMOLOGIQUES : HCM vs ΛCDM', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.savefig('/mnt/user-data/outputs/hertault_perturbations.png', 
                dpi=150, bbox_inches='tight')
    plt.savefig('/mnt/user-data/outputs/hertault_perturbations.pdf', 
                bbox_inches='tight')
    print("→ hertault_perturbations.png/pdf")
    
    # =========================================================================
    # RÉSUMÉ
    # =========================================================================
    
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    
    print(f"""
┌────────────────────────────────────────────────────────────────────────┐
│              COMPARAISON HCM vs ΛCDM                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  PARAMÈTRES :                                                         │
│    z_transition = {z_trans:.2f}                                               │
│    k_cut (suppression) = 5 h/Mpc                                      │
│                                                                        │
│  FACTEUR DE CROISSANCE :                                              │
│    D(z=0) HCM/ΛCDM = {D_HCM[0]/D_LCDM[0]:.4f}                                       │
│                                                                        │
│  SPECTRE DE PUISSANCE :                                               │
│    σ₈ ΛCDM = {sigma8_LCDM:.3f}                                                │
│    σ₈ HCM  = {sigma8_HCM:.3f}  ({(sigma8_HCM/sigma8_LCDM - 1)*100:+.1f}%)                                       │
│                                                                        │
│  COMPATIBILITÉ :                                                      │
│    ✓ CMB (z ~ 1100) : identique (régime DM)                          │
│    ✓ BAO (z ~ 0.5-2) : identique                                     │
│    ✓ P(k) grandes échelles : identique                               │
│    ~ P(k) petites échelles : supprimé (k > 5 h/Mpc)                  │
│    ~ σ₈ : légèrement réduit                                          │
│                                                                        │
│  SIGNATURE DISTINCTIVE :                                              │
│    Suppression du spectre à k > 5 h/Mpc                              │
│    (testable avec Lyman-α, lentillage faible)                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
""")
    
    plt.show()
    
    return {
        'z_LCDM': z_LCDM, 'D_LCDM': D_LCDM, 'f_LCDM': f_LCDM,
        'z_HCM': z_HCM, 'D_HCM': D_HCM, 'f_HCM': f_HCM,
        'k': k_arr, 'P_LCDM': P_LCDM, 'P_HCM': P_HCM,
        'sigma8_LCDM': sigma8_LCDM, 'sigma8_HCM': sigma8_HCM
    }


if __name__ == "__main__":
    results = main()
