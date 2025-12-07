/**
 * ============================================================================
 * DARK GEOMETRY - MODULE CLASS
 * ============================================================================
 * 
 * Implémentation du modèle Dark Geometry (DG) de Hugo Hertault pour CLASS.
 * 
 * Le Dark Boson φ_DG est le mode conforme de l'espace-temps avec une masse
 * effective dépendant de la densité locale :
 * 
 *   m²_eff(ρ) = (α* M_Pl)² [1 - (ρ/ρ_c)^β]
 * 
 * où :
 *   - α* ≃ 0.075 (couplage UV, Asymptotic Safety)
 *   - ρ_c ≡ ρ_DE (densité critique = densité d'énergie noire)
 *   - β = 2/3 (exposant, motivation holographique)
 * 
 * Régimes :
 *   - ρ > ρ_c : m² < 0 (tachyonique) → comportement Dark Matter (w ≈ 0)
 *   - ρ < ρ_c : m² > 0 (stable)     → comportement Dark Energy (w ≈ -1)
 * 
 * Auteur : Hugo Hertault
 * Date   : Décembre 2025
 * 
 * ============================================================================
 */

#ifndef __DARK_GEOMETRY__
#define __DARK_GEOMETRY__

/* Pour M_PI sur certains systèmes */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Définir M_PI si non défini */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * CONSTANTES PHYSIQUES
 * ============================================================================ */

/* Constantes fondamentales (SI) */
#define DG_C_SI          2.99792458e8      /* Vitesse de la lumière [m/s] */
#define DG_HBAR_SI       1.054571817e-34   /* Constante de Planck réduite [J·s] */
#define DG_G_SI          6.67430e-11       /* Constante gravitationnelle [m³/kg/s²] */
#define DG_MPC_SI        3.085677581e22    /* Mégaparsec [m] */
#define DG_EV_SI         1.602176634e-19   /* Électron-volt [J] */

/* Masse de Planck réduite */
#define DG_M_PL_SI       2.435e18          /* M_Pl = √(ℏc/8πG) [GeV] en unités naturelles */
#define DG_M_PL_KG       2.176434e-8       /* Masse de Planck [kg] */

/* ============================================================================
 * PARAMÈTRES DU MODÈLE DARK GEOMETRY
 * ============================================================================ */

/**
 * Structure contenant les paramètres du modèle Dark Geometry
 */
typedef struct {
    
    /* --- Paramètres fondamentaux (fixés par la théorie) --- */
    
    double alpha_star;       /**< Couplage universel α* ≃ 0.075 (Asymptotic Safety) */
    double rho_c;            /**< Densité critique ρ_c ≡ ρ_DE [kg/m³] */
    double beta;             /**< Exposant β = 2/3 (holographique) */
    
    /* --- Paramètres dérivés --- */
    
    double m0_squared;       /**< m₀² = (α* M_Pl)² [eV²] */
    double rho_c_eV4;        /**< ρ_c en [eV⁴] */
    double z_transition;     /**< Redshift de transition DM → DE */
    
    /* --- Paramètres de suppression P(k) --- */
    
    double k_suppression;    /**< Échelle de suppression k_s [h/Mpc] */
    double beta_suppression; /**< Pente de la suppression */
    double A_suppression;    /**< Amplitude de suppression (fraction) */
    
    /* --- Options numériques --- */
    
    int use_tachyonic;       /**< Activer le régime tachyonique complet */
    int smooth_transition;   /**< Transition lisse DM/DE */
    double transition_width; /**< Largeur de la transition (en unités de ρ/ρ_c) */
    
} DarkGeometryParams;


/* ============================================================================
 * STRUCTURE POUR LES PERTURBATIONS DU CHAMP SCALAIRE
 * ============================================================================ */

/**
 * Variables de perturbation pour le Dark Boson
 */
typedef struct {
    
    double delta_phi;        /**< Perturbation du champ δφ/φ̄ */
    double delta_phi_prime;  /**< Dérivée conforme δφ' */
    
    double delta_rho;        /**< Perturbation de densité δρ/ρ̄ */
    double theta;            /**< Divergence de vitesse θ */
    
    double pi_phi;           /**< Moment conjugué */
    
    /* Quantités dérivées */
    double w_eff;            /**< Équation d'état effective */
    double cs2_eff;          /**< Vitesse du son effective c_s² */
    double m2_eff;           /**< Masse effective m²_eff */
    
} DarkBosonPerturbations;


/* ============================================================================
 * FONCTIONS PRINCIPALES
 * ============================================================================ */

/* --- Initialisation et paramètres --- */

/**
 * Initialise les paramètres Dark Geometry avec les valeurs par défaut
 */
int dg_init_default(DarkGeometryParams *dg);

/**
 * Initialise les paramètres à partir des données cosmologiques
 */
int dg_init_from_cosmo(DarkGeometryParams *dg, 
                       double H0,        /* km/s/Mpc */
                       double Omega_m,
                       double Omega_DE);

/**
 * Calcule les quantités dérivées
 */
int dg_compute_derived(DarkGeometryParams *dg, double H0, double Omega_m);

/**
 * Affiche les paramètres
 */
void dg_print_params(const DarkGeometryParams *dg);


/* --- Masse effective --- */

/**
 * Calcule m²_eff(ρ) / m₀²
 * 
 * @param rho      Densité locale [kg/m³]
 * @param dg       Paramètres DG
 * @return         m²_eff / m₀² = 1 - (ρ/ρ_c)^β
 */
double dg_m2_eff_ratio(double rho, const DarkGeometryParams *dg);

/**
 * Calcule m²_eff(z) / m₀² en fonction du redshift
 */
double dg_m2_eff_ratio_z(double z, double Omega_m, double rho_crit_0,
                         const DarkGeometryParams *dg);

/**
 * Calcule m²_eff en unités absolues [eV²]
 */
double dg_m2_eff_eV2(double rho, const DarkGeometryParams *dg);


/* --- Équation d'état --- */

/**
 * Équation d'état effective w_φ(ρ)
 * 
 * - Régime DM (m² < 0) : w ≈ 0
 * - Régime DE (m² > 0) : w ≈ -1
 */
double dg_w_effective(double rho, const DarkGeometryParams *dg);

/**
 * Équation d'état en fonction de z
 */
double dg_w_effective_z(double z, double Omega_m, double rho_crit_0,
                        const DarkGeometryParams *dg);

/**
 * Vitesse du son effective c_s²
 */
double dg_cs2_effective(double rho, const DarkGeometryParams *dg);


/* --- Densité et pression --- */

/**
 * Densité d'énergie du champ φ
 */
double dg_rho_phi(double phi, double phi_dot, double m2_eff);

/**
 * Pression du champ φ
 */
double dg_p_phi(double phi, double phi_dot, double m2_eff);


/* --- Suppression du spectre de puissance --- */

/**
 * Fonction de suppression S(k) = P_DG(k) / P_ΛCDM(k)
 * 
 * @param k        Nombre d'onde [h/Mpc]
 * @param dg       Paramètres DG
 * @return         Facteur de suppression ∈ [1-A, 1]
 */
double dg_suppression(double k, const DarkGeometryParams *dg);

/**
 * Dérivée de la suppression dS/dk
 */
double dg_suppression_derivative(double k, const DarkGeometryParams *dg);


/* --- Échelles caractéristiques --- */

/**
 * Longueur de Jeans effective λ_J(ρ)
 */
double dg_jeans_length(double rho, const DarkGeometryParams *dg);

/**
 * Nombre d'onde de Jeans k_J = 2π/λ_J
 */
double dg_jeans_wavenumber(double rho, const DarkGeometryParams *dg);

/**
 * Redshift de transition DM → DE
 */
double dg_z_transition(double Omega_m, double rho_crit_0, 
                       const DarkGeometryParams *dg);


/* ============================================================================
 * ÉQUATIONS DE PERTURBATION (pour intégration dans CLASS)
 * ============================================================================ */

/**
 * Équation d'évolution du fond pour φ̄(τ)
 * 
 * φ̄'' + 2ℋφ̄' + a² m²_eff φ̄ = 0
 * 
 * @param a          Facteur d'échelle
 * @param H          Paramètre de Hubble H = ȧ/a
 * @param phi        Valeur du champ
 * @param phi_prime  Dérivée conforme φ'
 * @param m2_eff     Masse effective
 * @return           φ''
 */
double dg_background_eom(double a, double H, double phi, double phi_prime, 
                         double m2_eff);

/**
 * Équations de perturbation pour δφ (jauge synchrone)
 * 
 * δφ'' + 2ℋδφ' + (k² + a²m²_eff)δφ = (φ̄'/2)h' - a²(∂m²/∂ρ)δρ φ̄
 * 
 * @param k          Nombre d'onde
 * @param a          Facteur d'échelle
 * @param H          Paramètre de Hubble
 * @param pert       Structure de perturbations (entrée/sortie)
 * @param h_prime    Trace de la métrique h'
 * @param dg         Paramètres DG
 */
int dg_perturbation_eom(double k, double a, double H,
                        DarkBosonPerturbations *pert,
                        double h_prime, double phi_bar, double phi_bar_prime,
                        double rho_m, const DarkGeometryParams *dg);

/**
 * Calcule δρ et δp à partir de δφ
 */
int dg_compute_delta_rho_p(double a, double phi_bar, double phi_bar_prime,
                           const DarkBosonPerturbations *pert,
                           double m2_eff, double *delta_rho, double *delta_p);


/* ============================================================================
 * INTÉGRATION BACKGROUND (système d'équations)
 * ============================================================================ */

/**
 * Structure pour l'état du background
 */
typedef struct {
    double tau;          /**< Temps conforme */
    double a;            /**< Facteur d'échelle */
    double phi;          /**< Champ scalaire */
    double phi_prime;    /**< Dérivée conforme */
    double H;            /**< Paramètre de Hubble conforme ℋ = a'/a */
    double rho_phi;      /**< Densité d'énergie du champ */
    double p_phi;        /**< Pression du champ */
    double w_phi;        /**< Équation d'état */
    double Omega_phi;    /**< Fraction de densité du champ */
} DGBackgroundState;

/**
 * Dérivées pour l'intégration du background
 * 
 * @param state      État actuel
 * @param derivs     Dérivées (sortie)
 * @param dg         Paramètres DG
 * @param Omega_m0   Densité de matière aujourd'hui
 * @param Omega_r0   Densité de radiation aujourd'hui
 */
int dg_background_derivs(const DGBackgroundState *state, 
                         DGBackgroundState *derivs,
                         const DarkGeometryParams *dg,
                         double Omega_m0, double Omega_r0);


/* ============================================================================
 * UTILITAIRES
 * ============================================================================ */

/**
 * Conversion ρ [kg/m³] → ρ [eV⁴]
 */
double dg_rho_to_eV4(double rho_kg_m3);

/**
 * Conversion ρ [eV⁴] → ρ [kg/m³]
 */
double dg_rho_from_eV4(double rho_eV4);

/**
 * Calcule ρ_m(z) = ρ_m0 (1+z)³
 */
double dg_rho_matter(double z, double Omega_m, double rho_crit_0);


#endif /* __DARK_GEOMETRY__ */
