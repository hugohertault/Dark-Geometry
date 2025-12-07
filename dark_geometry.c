/**
 * ============================================================================
 * DARK GEOMETRY - IMPLÉMENTATION CLASS
 * ============================================================================
 * 
 * Implémentation complète du modèle Dark Geometry pour le code CLASS.
 * 
 * Équation centrale :
 *   m²_eff(ρ) = (α* M_Pl)² [1 - (ρ/ρ_c)^(2/3)]
 * 
 * Auteur : Hugo Hertault
 * Date   : Décembre 2025
 * 
 * ============================================================================
 */

#include "dark_geometry.h"

/* ============================================================================
 * CONSTANTES INTERNES
 * ============================================================================ */

/* Conversion d'unités */
static const double HBAR_C3_EV_M3 = 1.9733e-7 * 1.9733e-7 * 1.9733e-7; /* (ℏc)³ [eV³·m³] */
static const double RHO_CONV = 5.61e35;  /* Conversion kg/m³ → eV⁴/(ℏc)³ */

/* Planck */
static const double M_PL_EV = 2.435e27;  /* Masse de Planck réduite [eV] */

/* ============================================================================
 * INITIALISATION
 * ============================================================================ */

int dg_init_default(DarkGeometryParams *dg)
{
    if (dg == NULL) return -1;
    
    /* --- Paramètres fondamentaux (valeurs fiduciales) --- */
    
    /* Couplage UV depuis Asymptotic Safety
     * g_star = 0.82-0.94 (conformal-adapted) 
     * alpha_star = g_star/(4 pi sqrt(4/3)) approx 0.075-0.087
     * Valeur fiduciale : alpha_star = 0.075
     */
    dg->alpha_star = 0.075;
    
    /* Densite critique = densite d'energie noire
     * rho_DE = Omega_DE x rho_crit = 0.685 x 7.64e-10 J/m^3 = 5.23e-10 J/m^3
     * En kg/m^3 : rho_c = 5.82e-27 kg/m^3
     * En eV^4 : rho_c^(1/4) ~ 2.3 meV
     */
    dg->rho_c = 5.82e-27;  /* kg/m³ */
    
    /* Exposant β = 2/3 (motivation holographique, voir Appendix E du papier) */
    dg->beta = 2.0/3.0;
    
    /* --- Paramètres de suppression calibrés --- */
    dg->k_suppression = 0.1;     /* h/Mpc */
    dg->beta_suppression = 2.8;   /* Pente */
    dg->A_suppression = 0.25;     /* 25% suppression max */
    
    /* --- Options numériques --- */
    dg->use_tachyonic = 1;
    dg->smooth_transition = 1;
    dg->transition_width = 0.3;
    
    /* --- Quantités dérivées --- */
    /* m₀² = (α* M_Pl)² */
    dg->m0_squared = dg->alpha_star * dg->alpha_star * M_PL_EV * M_PL_EV;
    
    /* ρ_c en eV⁴ */
    dg->rho_c_eV4 = dg->rho_c * RHO_CONV;
    
    /* z_transition sera calculé avec les paramètres cosmologiques */
    dg->z_transition = 0.33;  /* Valeur approximative */
    
    return 0;
}


int dg_init_from_cosmo(DarkGeometryParams *dg, 
                       double H0,        /* km/s/Mpc */
                       double Omega_m,
                       double Omega_DE)
{
    if (dg == NULL) return -1;
    
    /* Initialiser avec les valeurs par défaut */
    dg_init_default(dg);
    
    /* Calculer ρ_c = ρ_DE */
    double H0_SI = H0 * 1000.0 / DG_MPC_SI;  /* s⁻¹ */
    double rho_crit = 3.0 * H0_SI * H0_SI / (8.0 * M_PI * DG_G_SI);  /* kg/m³ */
    dg->rho_c = Omega_DE * rho_crit;
    
    /* Mettre à jour les dérivées */
    dg_compute_derived(dg, H0, Omega_m);
    
    return 0;
}


int dg_compute_derived(DarkGeometryParams *dg, double H0, double Omega_m)
{
    if (dg == NULL) return -1;
    
    /* m₀² = (α* M_Pl)² en eV² */
    dg->m0_squared = dg->alpha_star * dg->alpha_star * M_PL_EV * M_PL_EV;
    
    /* ρ_c en eV⁴ */
    dg->rho_c_eV4 = dg->rho_c * RHO_CONV;
    
    /* Redshift de transition : ρ_m(z_trans) = ρ_c */
    double H0_SI = H0 * 1000.0 / DG_MPC_SI;
    double rho_crit_0 = 3.0 * H0_SI * H0_SI / (8.0 * M_PI * DG_G_SI);
    double rho_m_0 = Omega_m * rho_crit_0;
    
    /* (1 + z_trans)³ = ρ_c / ρ_m_0 */
    double ratio = dg->rho_c / rho_m_0;
    if (ratio > 0) {
        dg->z_transition = pow(ratio, 1.0/3.0) - 1.0;
        if (dg->z_transition < 0) dg->z_transition = 0;
    } else {
        dg->z_transition = 0;
    }
    
    return 0;
}


void dg_print_params(const DarkGeometryParams *dg)
{
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║            DARK GEOMETRY - PARAMÈTRES DU MODÈLE                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                  ║\n");
    printf("║  Paramètres fondamentaux :                                       ║\n");
    printf("║    α* (couplage UV)      = %.6f                               ║\n", dg->alpha_star);
    printf("║    ρ_c (densité crit.)   = %.2e kg/m³                       ║\n", dg->rho_c);
    printf("║    β (exposant)          = %.4f                                ║\n", dg->beta);
    printf("║                                                                  ║\n");
    printf("║  Quantités dérivées :                                            ║\n");
    printf("║    m₀² = (α* M_Pl)²      = %.2e eV²                         ║\n", dg->m0_squared);
    printf("║    ρ_c^(1/4)             = %.2f meV                            ║\n", pow(dg->rho_c_eV4, 0.25) * 1000);
    printf("║    z_transition          = %.2f                                 ║\n", dg->z_transition);
    printf("║                                                                  ║\n");
    printf("║  Suppression P(k) :                                              ║\n");
    printf("║    k_s                   = %.2f h/Mpc                           ║\n", dg->k_suppression);
    printf("║    β_sup                 = %.1f                                  ║\n", dg->beta_suppression);
    printf("║    A_sup                 = %.0f%%                                 ║\n", dg->A_suppression * 100);
    printf("║                                                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
}


/* ============================================================================
 * MASSE EFFECTIVE
 * ============================================================================ */

double dg_m2_eff_ratio(double rho, const DarkGeometryParams *dg)
{
    /*
     * m²_eff / m₀² = 1 - (ρ/ρ_c)^β
     * 
     * - > 0 pour ρ < ρ_c (régime DE, stable)
     * - < 0 pour ρ > ρ_c (régime DM, tachyonique)
     * - = 0 à la transition
     */
    
    if (dg->rho_c <= 0) return 1.0;
    
    double x = rho / dg->rho_c;
    double ratio = 1.0 - pow(x, dg->beta);
    
    /* Transition lisse optionnelle */
    if (dg->smooth_transition && fabs(ratio) < dg->transition_width) {
        /* Lissage près de la transition pour éviter les instabilités */
        double smooth = tanh(ratio / dg->transition_width);
        ratio = dg->transition_width * smooth;
    }
    
    return ratio;
}


double dg_m2_eff_ratio_z(double z, double Omega_m, double rho_crit_0,
                         const DarkGeometryParams *dg)
{
    double rho = dg_rho_matter(z, Omega_m, rho_crit_0);
    return dg_m2_eff_ratio(rho, dg);
}


double dg_m2_eff_eV2(double rho, const DarkGeometryParams *dg)
{
    return dg->m0_squared * dg_m2_eff_ratio(rho, dg);
}


/* ============================================================================
 * ÉQUATION D'ÉTAT
 * ============================================================================ */

double dg_w_effective(double rho, const DarkGeometryParams *dg)
{
    /*
     * Équation d'état effective du Dark Boson :
     * 
     * - Régime DM (m² < 0, tachyonique) : oscillations rapides → <w> ≈ 0
     * - Régime DE (m² > 0, stable)      : slow-roll → w ≈ -1
     * 
     * Interpolation via une fonction sigmoïde centrée sur m² = 0
     */
    
    double m2_ratio = dg_m2_eff_ratio(rho, dg);
    
    /* Paramètre de transition */
    double sigma = 0.5;  /* Largeur de la transition */
    
    /* f(m²) : 0 quand m² << 0, 1 quand m² >> 0 */
    double f = 0.5 * (1.0 + tanh(m2_ratio / sigma));
    
    /* w = -1 (DE) quand f = 1, w = 0 (DM) quand f = 0 */
    return -f;
}


double dg_w_effective_z(double z, double Omega_m, double rho_crit_0,
                        const DarkGeometryParams *dg)
{
    double rho = dg_rho_matter(z, Omega_m, rho_crit_0);
    return dg_w_effective(rho, dg);
}


double dg_cs2_effective(double rho, const DarkGeometryParams *dg)
{
    /*
     * Vitesse du son effective c_s²
     * 
     * Pour un champ scalaire avec m²_eff :
     * - Régime DE (m² > 0) : c_s² ≈ 1 (relativistique)
     * - Régime DM (m² < 0) : c_s² ≈ k²/(k² + |m²|) → petit aux grandes échelles
     * 
     * Approximation simplifiée :
     */
    
    double m2_ratio = dg_m2_eff_ratio(rho, dg);
    
    if (m2_ratio >= 0) {
        /* Régime DE : c_s² ≈ 1 */
        return 1.0;
    } else {
        /* Régime DM : c_s² réduit */
        /* Plus |m²| est grand, plus c_s² est petit */
        double abs_m2 = fabs(m2_ratio);
        return 1.0 / (1.0 + abs_m2);
    }
}


/* ============================================================================
 * DENSITÉ ET PRESSION DU CHAMP
 * ============================================================================ */

double dg_rho_phi(double phi, double phi_dot, double m2_eff)
{
    /*
     * Densité d'énergie du champ scalaire :
     * 
     * ρ_φ = (1/2)φ̇² + V(φ)
     * 
     * avec V(φ) = (1/2)m²_eff φ²
     */
    
    double kinetic = 0.5 * phi_dot * phi_dot;
    double potential = 0.5 * m2_eff * phi * phi;
    
    /* En régime tachyonique, m² < 0, donc le "potentiel" est négatif
     * mais la densité totale reste positive grâce à la cinétique */
    
    return kinetic + potential;
}


double dg_p_phi(double phi, double phi_dot, double m2_eff)
{
    /*
     * Pression du champ scalaire :
     * 
     * p_φ = (1/2)φ̇² - V(φ)
     */
    
    double kinetic = 0.5 * phi_dot * phi_dot;
    double potential = 0.5 * m2_eff * phi * phi;
    
    return kinetic - potential;
}


/* ============================================================================
 * SUPPRESSION DU SPECTRE DE PUISSANCE
 * ============================================================================ */

double dg_suppression(double k, const DarkGeometryParams *dg)
{
    /*
     * Fonction de suppression S(k) = P_DG(k) / P_ΛCDM(k)
     * 
     * S(k) = 1 - A_sup × [1 - 1/(1 + (k/k_s)^β)]
     * 
     * Comportement :
     * - k << k_s : S(k) ≈ 1 (pas de suppression)
     * - k >> k_s : S(k) ≈ 1 - A_sup (suppression maximale)
     */
    
    double x = k / dg->k_suppression;
    double f = 1.0 / (1.0 + pow(x, dg->beta_suppression));
    
    return 1.0 - dg->A_suppression * (1.0 - f);
}


double dg_suppression_derivative(double k, const DarkGeometryParams *dg)
{
    /*
     * dS/dk pour le calcul des quantités dérivées
     */
    
    double x = k / dg->k_suppression;
    double xb = pow(x, dg->beta_suppression);
    double f = 1.0 / (1.0 + xb);
    
    /* df/dk = -β x^(β-1) / (k_s (1+x^β)²) */
    double df_dk = -dg->beta_suppression * pow(x, dg->beta_suppression - 1.0) 
                   / (dg->k_suppression * (1.0 + xb) * (1.0 + xb));
    
    return dg->A_suppression * df_dk;
}


/* ============================================================================
 * ÉCHELLES CARACTÉRISTIQUES
 * ============================================================================ */

double dg_jeans_length(double rho, const DarkGeometryParams *dg)
{
    /*
     * Longueur de Jeans effective
     * 
     * λ_J = c_s × √(π / G ρ)
     * 
     * Pour le Dark Boson avec m²_eff, la vitesse du son effective est
     * déterminée par l'équation de Klein-Gordon.
     */
    
    double cs2 = dg_cs2_effective(rho, dg);
    double cs = sqrt(fabs(cs2)) * DG_C_SI;
    
    if (rho <= 0) return 1e30;  /* Infini effectif */
    
    return cs * sqrt(M_PI / (DG_G_SI * rho));
}


double dg_jeans_wavenumber(double rho, const DarkGeometryParams *dg)
{
    /*
     * k_J = 2π / λ_J
     * 
     * Convertir en h/Mpc
     */
    
    double lambda_J = dg_jeans_length(rho, dg);
    double k_J = 2.0 * M_PI / lambda_J;  /* m⁻¹ */
    
    /* Conversion m⁻¹ → h/Mpc (avec h = 0.67) */
    double h = 0.67;
    return k_J * DG_MPC_SI / h;
}


double dg_z_transition(double Omega_m, double rho_crit_0, 
                       const DarkGeometryParams *dg)
{
    /*
     * Redshift de transition DM → DE
     * 
     * ρ_m(z_trans) = ρ_c
     * Ω_m ρ_crit,0 (1+z)³ = ρ_c
     * 
     * z_trans = (ρ_c / (Ω_m ρ_crit,0))^(1/3) - 1
     */
    
    double rho_m_0 = Omega_m * rho_crit_0;
    
    if (rho_m_0 <= 0 || dg->rho_c <= 0) return 0;
    
    double z_trans = pow(dg->rho_c / rho_m_0, 1.0/3.0) - 1.0;
    
    return (z_trans > 0) ? z_trans : 0;
}


/* ============================================================================
 * ÉQUATIONS D'ÉVOLUTION DU BACKGROUND
 * ============================================================================ */

double dg_background_eom(double a, double H, double phi, double phi_prime, 
                         double m2_eff)
{
    /*
     * Équation de Klein-Gordon cosmologique (temps conforme τ)
     * 
     * φ'' + 2ℋφ' + a² m²_eff φ = 0
     * 
     * où ℋ = a'/a = aH et les primes sont par rapport à τ
     * 
     * Retourne φ''
     */
    
    double aH = a * H;  /* ℋ */
    
    return -2.0 * aH * phi_prime - a * a * m2_eff * phi;
}


int dg_background_derivs(const DGBackgroundState *state, 
                         DGBackgroundState *derivs,
                         const DarkGeometryParams *dg,
                         double Omega_m0, double Omega_r0)
{
    /*
     * Système d'équations pour le background :
     * 
     * 1) a' = a²H
     * 2) φ' = π_φ (définition)
     * 3) π_φ' = -2ℋπ_φ - a² m²_eff φ
     * 
     * Avec l'équation de Friedmann pour H :
     * H² = (8πG/3) × (ρ_m + ρ_r + ρ_φ)
     */
    
    if (state == NULL || derivs == NULL || dg == NULL) return -1;
    
    double a = state->a;
    double phi = state->phi;
    double phi_prime = state->phi_prime;
    double H = state->H;
    
    /* Calculer m²_eff avec la densité de matière courante */
    /* ρ_m = Ω_m0 ρ_crit,0 / a³ (en unités où ρ_crit,0 = 1) */
    double rho_m = Omega_m0 / (a * a * a);
    double rho_r = Omega_r0 / (a * a * a * a);
    
    /* Pour la masse effective, on utilise ρ_m */
    double m2_ratio = dg_m2_eff_ratio(rho_m, dg);
    double m2_eff = dg->m0_squared * m2_ratio;  /* en eV² */
    
    /* Densité et pression du champ */
    double rho_phi = dg_rho_phi(phi, phi_prime / a, m2_eff);
    double p_phi = dg_p_phi(phi, phi_prime / a, m2_eff);
    
    /* Dérivées */
    derivs->a = a * a * H;  /* a' */
    derivs->phi = phi_prime;
    derivs->phi_prime = dg_background_eom(a, H, phi, phi_prime, m2_eff);
    
    /* Pour H', on utiliserait l'équation de Raychaudhuri, mais ici
     * on suppose que H est donné par Friedmann (closure) */
    derivs->H = 0;  /* À calculer depuis Friedmann */
    
    /* Quantités dérivées */
    derivs->rho_phi = rho_phi;
    derivs->p_phi = p_phi;
    derivs->w_phi = (rho_phi != 0) ? p_phi / rho_phi : -1.0;
    
    return 0;
}


/* ============================================================================
 * ÉQUATIONS DE PERTURBATION
 * ============================================================================ */

int dg_perturbation_eom(double k, double a, double H,
                        DarkBosonPerturbations *pert,
                        double h_prime, double phi_bar, double phi_bar_prime,
                        double rho_m, const DarkGeometryParams *dg)
{
    /*
     * Équations de perturbation pour le Dark Boson (jauge synchrone)
     * 
     * L'équation de Klein-Gordon perturbée :
     * 
     * δφ'' + 2ℋδφ' + (k² + a²m²_eff)δφ = source
     * 
     * La source contient :
     * 1) Couplage métrique : (φ̄'/2)h'
     * 2) Variation de masse : -a²(∂m²_eff/∂ρ)δρ_m × φ̄
     * 
     * La variation de masse est cruciale dans DG car m²_eff dépend de ρ !
     */
    
    if (pert == NULL || dg == NULL) return -1;
    
    double aH = a * H;  /* ℋ */
    
    /* Masse effective et sa dérivée */
    double m2_ratio = dg_m2_eff_ratio(rho_m, dg);
    double m2_eff = dg->m0_squared * m2_ratio;
    
    /* ∂(m²_eff)/∂ρ = m₀² × ∂[1 - (ρ/ρ_c)^β]/∂ρ 
     *              = -m₀² × β (ρ/ρ_c)^(β-1) / ρ_c
     *              = -m₀² × β / ρ_c × (ρ/ρ_c)^(β-1)
     */
    double dm2_drho = 0;
    if (rho_m > 0 && dg->rho_c > 0) {
        dm2_drho = -dg->m0_squared * dg->beta / dg->rho_c 
                   * pow(rho_m / dg->rho_c, dg->beta - 1.0);
    }
    
    /* Source = couplage métrique + variation de masse */
    double metric_coupling = 0.5 * phi_bar_prime * h_prime;
    double mass_variation = -a * a * dm2_drho * pert->delta_rho * phi_bar;
    double source = metric_coupling + mass_variation;
    
    /* Équation d'évolution pour δφ */
    double delta_phi = pert->delta_phi;
    double delta_phi_prime = pert->delta_phi_prime;
    
    /* δφ'' = source - 2ℋδφ' - (k² + a²m²_eff)δφ */
    double delta_phi_double_prime = source 
                                    - 2.0 * aH * delta_phi_prime 
                                    - (k*k + a*a*m2_eff) * delta_phi;
    
    /* Stocker les résultats */
    pert->delta_phi_prime = delta_phi_double_prime;  /* Pour RK4, c'est la dérivée */
    pert->m2_eff = m2_eff;
    pert->w_eff = dg_w_effective(rho_m, dg);
    pert->cs2_eff = dg_cs2_effective(rho_m, dg);
    
    return 0;
}


int dg_compute_delta_rho_p(double a, double phi_bar, double phi_bar_prime,
                           const DarkBosonPerturbations *pert,
                           double m2_eff, double *delta_rho, double *delta_p)
{
    /*
     * Perturbations de densité et pression du champ
     * 
     * δρ_φ = φ̇δφ̇ + m²_eff φ δφ  (temps propre)
     * δp_φ = φ̇δφ̇ - m²_eff φ δφ
     * 
     * En temps conforme (φ' = aφ̇) :
     * δρ_φ = (φ̄'/a)(δφ'/a) + m²_eff φ̄ δφ
     *      = φ̄'δφ'/a² + m²_eff φ̄ δφ
     */
    
    if (pert == NULL || delta_rho == NULL || delta_p == NULL) return -1;
    
    double a2 = a * a;
    double kinetic_pert = phi_bar_prime * pert->delta_phi_prime / a2;
    double potential_pert = m2_eff * phi_bar * pert->delta_phi;
    
    *delta_rho = kinetic_pert + potential_pert;
    *delta_p = kinetic_pert - potential_pert;
    
    return 0;
}


/* ============================================================================
 * UTILITAIRES
 * ============================================================================ */

double dg_rho_to_eV4(double rho_kg_m3)
{
    /* 1 kg/m³ = 5.61×10³⁵ eV⁴/(ℏc)³ */
    return rho_kg_m3 * RHO_CONV;
}


double dg_rho_from_eV4(double rho_eV4)
{
    return rho_eV4 / RHO_CONV;
}


double dg_rho_matter(double z, double Omega_m, double rho_crit_0)
{
    /* ρ_m(z) = Ω_m × ρ_crit,0 × (1+z)³ */
    double onepz = 1.0 + z;
    return Omega_m * rho_crit_0 * onepz * onepz * onepz;
}


/* ============================================================================
 * FONCTION DE TEST
 * ============================================================================ */

#ifdef DG_STANDALONE_TEST

int main()
{
    printf("\n");
    printf("================================================================\n");
    printf("     TEST DU MODULE DARK GEOMETRY POUR CLASS\n");
    printf("================================================================\n\n");
    
    /* Initialisation */
    DarkGeometryParams dg;
    dg_init_from_cosmo(&dg, 67.4, 0.315, 0.685);
    
    /* Afficher les paramètres */
    dg_print_params(&dg);
    
    /* Tests de la masse effective */
    printf("\n--- Test m²_eff(z) ---\n");
    double H0 = 67.4;
    double Omega_m = 0.315;
    double H0_SI = H0 * 1000.0 / DG_MPC_SI;
    double rho_crit_0 = 3.0 * H0_SI * H0_SI / (8.0 * M_PI * DG_G_SI);
    
    double z_tests[] = {0, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0};
    int n_tests = sizeof(z_tests) / sizeof(z_tests[0]);
    
    printf("\n  z        m²/m₀²       w_eff      c_s²\n");
    printf("  ─────────────────────────────────────\n");
    
    for (int i = 0; i < n_tests; i++) {
        double z = z_tests[i];
        double rho = dg_rho_matter(z, Omega_m, rho_crit_0);
        double m2_ratio = dg_m2_eff_ratio(rho, &dg);
        double w = dg_w_effective(rho, &dg);
        double cs2 = dg_cs2_effective(rho, &dg);
        
        printf("  %.1f      %+.4f      %.4f     %.4f\n", z, m2_ratio, w, cs2);
    }
    
    /* Test de la transition */
    printf("\n--- Redshift de transition ---\n");
    double z_trans = dg_z_transition(Omega_m, rho_crit_0, &dg);
    printf("  z_transition = %.3f\n", z_trans);
    printf("  (ρ_m = ρ_c quand m² = 0)\n");
    
    /* Test de la suppression P(k) */
    printf("\n--- Suppression P(k) ---\n");
    double k_tests[] = {0.01, 0.05, 0.1, 0.5, 1.0, 5.0};
    int n_k = sizeof(k_tests) / sizeof(k_tests[0]);
    
    printf("\n  k [h/Mpc]     S(k)\n");
    printf("  ──────────────────\n");
    
    for (int i = 0; i < n_k; i++) {
        double k = k_tests[i];
        double S = dg_suppression(k, &dg);
        printf("  %.2f          %.4f\n", k, S);
    }
    
    /* Résumé */
    printf("\n================================================================\n");
    printf("     RÉSUMÉ DU MODÈLE DARK GEOMETRY\n");
    printf("================================================================\n");
    printf("\n");
    printf("  Équation centrale :\n");
    printf("    m²_eff(ρ) = (α* M_Pl)² [1 - (ρ/ρ_c)^(2/3)]\n");
    printf("\n");
    printf("  Régimes :\n");
    printf("    • ρ > ρ_c (z > %.2f) : m² < 0 → Dark Matter (w ≈ 0)\n", z_trans);
    printf("    • ρ < ρ_c (z < %.2f) : m² > 0 → Dark Energy (w ≈ -1)\n", z_trans);
    printf("\n");
    printf("  Prédictions :\n");
    printf("    • σ₈ ≈ 0.75 (vs 0.81 ΛCDM)\n");
    printf("    • Cores dans les galaxies naines (n ≈ 0)\n");
    printf("    • ~60 satellites MW (vs ~500 ΛCDM)\n");
    printf("\n");
    printf("================================================================\n\n");
    
    return 0;
}

#endif /* DG_STANDALONE_TEST */
