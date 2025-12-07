# Dérivation Rigoureuse du Système Couplé Hertault

## 1. Point de départ : Action du champ scalaire

L'action totale est :

$$S = S_{\text{grav}} + S_\phi + S_m + S_{\text{int}}$$

avec :

$$S_\phi = \int d^4x \sqrt{-g} \left[ -\frac{1}{2} g^{\mu\nu} \partial_\mu \phi \partial_\nu \phi - V(\phi) \right]$$

$$S_{\text{int}} = \int d^4x \sqrt{-g} \left[ -\frac{\alpha^*}{M_{\text{Pl}}} \phi \, T^\mu_\mu \right]$$

## 2. Équation du mouvement

La variation δS/δφ = 0 donne :

$$\Box \phi - \frac{dV}{d\phi} = \frac{\alpha^*}{M_{\text{Pl}}} T^\mu_\mu$$

où $\Box = \frac{1}{\sqrt{-g}} \partial_\mu (\sqrt{-g} g^{\mu\nu} \partial_\nu)$

## 3. Potentiel effectif HCM

Le potentiel du champ de Hertault dépend de la densité locale :

$$V(\phi, \rho) = \frac{1}{2} m^2_{\text{eff}}(\rho) \phi^2$$

avec :

$$m^2_{\text{eff}}(\rho) = m_0^2 \left[ 1 - \left(\frac{\rho}{\rho_c}\right)^{2/3} \right]$$

où $m_0 = \alpha^* M_{\text{Pl}}$.

## 4. Régime statique, symétrie sphérique

### 4.1 Métrique

En approximation de champ faible (Newtonien) :

$$ds^2 = -(1 + 2\Phi/c^2) c^2 dt^2 + (1 - 2\Phi/c^2)(dr^2 + r^2 d\Omega^2)$$

où Φ est le potentiel gravitationnel Newtonien.

### 4.2 Équation de Klein-Gordon

Pour un champ statique φ = φ(r), le d'Alembertien se réduit au Laplacien :

$$\Box \phi \approx \nabla^2 \phi = \frac{1}{r^2} \frac{d}{dr}\left( r^2 \frac{d\phi}{dr} \right)$$

### 4.3 Source

Pour de la matière non-relativiste : $T^\mu_\mu = -\rho_m c^2$

(Convention : signature (-,+,+,+), donc $T^0_0 = -\rho c^2$, $T^i_i = p \approx 0$)

### 4.4 Équation complète

$$\frac{1}{r^2} \frac{d}{dr}\left( r^2 \frac{d\phi}{dr} \right) - m^2_{\text{eff}}(\rho) \phi = -\frac{\alpha^*}{M_{\text{Pl}}} \rho_m c^2$$

**Note sur les signes** : Le signe moins devant le terme source vient de $T^\mu_\mu = -\rho_m c^2$.

## 5. Système d'équations couplées

### 5.1 Variables

- φ(r) : champ scalaire
- ψ(r) = dφ/dr : dérivée du champ
- ρ_m(r) : densité de matière baryonique (donnée)
- ρ_φ(r) : densité d'énergie du champ (à calculer)
- ρ_total(r) = ρ_m(r) + ρ_φ(r)

### 5.2 Densité d'énergie du champ

Pour un champ statique :

$$\rho_\phi = \frac{1}{2} \left(\frac{d\phi}{dr}\right)^2 + V(\phi)$$

$$\rho_\phi = \frac{1}{2} \psi^2 + \frac{1}{2} m^2_{\text{eff}}(\rho_{\text{total}}) \phi^2$$

**Attention** : Dans le régime tachyonique où $m^2_{\text{eff}} < 0$, le terme de potentiel est négatif.
Pour la densité d'énergie physique, on prend :

$$\rho_\phi = \frac{1}{2} \psi^2 + \frac{1}{2} |m^2_{\text{eff}}| \phi^2$$

### 5.3 Système différentiel

$$\begin{cases}
\dfrac{d\phi}{dr} = \psi \\[12pt]
\dfrac{d\psi}{dr} = -\dfrac{2}{r} \psi + m^2_{\text{eff}}(\rho_{\text{total}}) \phi - \dfrac{\alpha^*}{M_{\text{Pl}}} \rho_m c^2 \\[12pt]
\rho_{\text{total}} = \rho_m + \rho_\phi \\[12pt]
\rho_\phi = \dfrac{1}{2} \psi^2 + \dfrac{1}{2} |m^2_{\text{eff}}| \phi^2 \\[12pt]
m^2_{\text{eff}} = m_0^2 \left[ 1 - \left(\dfrac{\rho_{\text{total}}}{\rho_c}\right)^{2/3} \right]
\end{cases}$$

## 6. Analyse dimensionnelle

### 6.1 Dimensions des quantités

| Quantité | Dimension SI | Unité naturelle (ℏ=c=1) |
|----------|--------------|-------------------------|
| φ | [M L² T⁻²]^(1/2) = [E]^(1/2) | [E] |
| m_eff | [T⁻¹] | [E] |
| ρ | [M L⁻³] | [E⁴] |
| α* | sans dimension | sans dimension |
| M_Pl | [M] | [E] |

### 6.2 Vérification de l'équation

Terme par terme dans l'équation de KG (unités SI) :

- $\nabla^2 \phi$ : [φ]/[L²] = [E]^(1/2) / [L²]
- $m^2_{\text{eff}} \phi$ : [T⁻²][E]^(1/2) = [E]^(1/2) / [L²] ✓
- $(\alpha^*/M_{\text{Pl}}) \rho_m c^2$ : [1]/[M] × [M L⁻³] × [L² T⁻²] = [L⁻²] × [E]^(1/2) ... 

**Problème** : Les dimensions ne matchent pas directement en SI.

### 6.3 Formulation correcte en unités naturelles

En unités naturelles (ℏ = c = 1) :

- [φ] = [E] = [M]
- [m] = [E] = [L⁻¹]
- [ρ] = [E⁴] = [M⁴] = [L⁻⁴]
- [M_Pl] = [E] = [M]

L'équation devient :

$$\nabla^2 \phi - m^2_{\text{eff}} \phi = \frac{\alpha^*}{M_{\text{Pl}}} \rho_m$$

Vérification :
- $\nabla^2 \phi$ : [E]/[L²] = [E³]
- $m^2 \phi$ : [E²][E] = [E³] ✓
- $(\alpha^*/M_{\text{Pl}}) \rho$ : [1]/[E] × [E⁴] = [E³] ✓

## 7. Passage aux unités SI

Pour coder en SI, on réécrit :

$$\nabla^2 \phi - \frac{m^2_{\text{eff}} c^2}{\hbar^2} \phi = \frac{\alpha^*}{M_{\text{Pl}} c^2} \rho_m c^2$$

où maintenant :
- φ a les dimensions de [E] (ou [M c²])
- m_eff est en [s⁻¹] ou équivalent eV/ℏ

**Simplification** : Définissons $\tilde{m}^2 = m^2 c^2/\hbar^2$ avec dimension [L⁻²]

$$\nabla^2 \phi - \tilde{m}^2_{\text{eff}} \phi = \frac{\alpha^*}{M_{\text{Pl}} c^2} \rho_m c^2$$

## 8. Normalisation pour le calcul numérique

### 8.1 Échelles caractéristiques

- Longueur : $r_0 = 1$ kpc $= 3.086 \times 10^{19}$ m
- Masse : $M_0 = 10^{10} M_\odot = 1.989 \times 10^{40}$ kg
- Densité : $\rho_0 = M_0 / r_0^3 = 6.77 \times 10^{-19}$ kg/m³
- Champ : $\phi_0$ = à déterminer

### 8.2 Variables sans dimension

$$\tilde{r} = r/r_0, \quad \tilde{\rho} = \rho/\rho_0, \quad \tilde{\phi} = \phi/\phi_0$$

### 8.3 Équation normalisée

Après normalisation, l'équation devient :

$$\frac{1}{\tilde{r}^2} \frac{d}{d\tilde{r}}\left( \tilde{r}^2 \frac{d\tilde{\phi}}{d\tilde{r}} \right) - \lambda^2 \tilde{m}^2_{\text{eff}} \tilde{\phi} = \beta \tilde{\rho}_m$$

où :
- $\lambda = r_0 \tilde{m}_0$ : nombre d'onde caractéristique × rayon
- $\beta = (\alpha^* r_0^2 \rho_0 c^2) / (M_{\text{Pl}} c^2 \phi_0)$ : couplage normalisé

## 9. Estimation des paramètres

### 9.1 Masse caractéristique du champ

$$m_0 = \alpha^* M_{\text{Pl}} = 0.075 \times 1.22 \times 10^{19} \text{ GeV} \approx 9 \times 10^{17} \text{ GeV}$$

En longueur de Compton :

$$\lambda_C = \frac{\hbar}{m_0 c} = \frac{1.055 \times 10^{-34}}{9 \times 10^{17} \times 1.6 \times 10^{-10} / (3 \times 10^8)^2} \approx 2 \times 10^{-34} \text{ m}$$

**C'est à l'échelle de Planck** — les oscillations sont bien sub-microscopiques.

### 9.2 Régime effectif

Puisque λ_C << r_0, on est dans le régime WKB où :

$$\lambda = r_0 / \lambda_C \approx 10^{53} >> 1$$

Cela signifie que la solution oscille extrêmement rapidement et doit être moyennée.

## 10. Conclusion sur la méthode numérique

La résolution directe du système couplé est **numériquement impossible** à cause de :
1. Oscillations à λ ~ 10⁻³⁴ m
2. Pas de temps/espace requis : Δr << λ

**Solution** : Utiliser l'approximation WKB et résoudre pour les **enveloppes** des oscillations.

C'est l'objet de l'Étape 2.
