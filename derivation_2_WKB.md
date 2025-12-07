# Dérivation WKB du Profil de Halo de Hertault

## 1. Problème à résoudre

L'équation de Klein-Gordon en symétrie sphérique :

$$\frac{d^2\phi}{dr^2} + \frac{2}{r}\frac{d\phi}{dr} + \mu^2(r) \phi = S(r)$$

où :
- $\mu^2(r) = -m^2_{\text{eff}}(r) > 0$ dans le régime tachyonique (ρ > ρc)
- $S(r) = -(\alpha^*/M_{\text{Pl}}) \rho_m(r) c^2$ est le terme source

**Paramètre clé** : $\mu r >> 1$ (régime WKB valide)

## 2. Approximation WKB standard

### 2.1 Ansatz

On cherche une solution de la forme :

$$\phi(r) = A(r) e^{i\theta(r)} + A^*(r) e^{-i\theta(r)}$$

où A(r) est une amplitude lentement variable et θ(r) une phase rapidement variable.

### 2.2 Condition WKB

L'approximation est valide si :

$$\left| \frac{d\ln A}{dr} \right| << \left| \frac{d\theta}{dr} \right| = \mu(r)$$

et

$$\left| \frac{d\mu}{dr} \right| << \mu^2$$

### 2.3 Solution à l'ordre dominant

À l'ordre dominant en WKB, la phase satisfait :

$$\frac{d\theta}{dr} = \mu(r)$$

donc :

$$\theta(r) = \int^r \mu(r') dr'$$

L'amplitude satisfait (conservation du flux) :

$$\frac{d}{dr}(r^2 A^2 \mu) = 0$$

donc :

$$A(r) = \frac{A_0}{\sqrt{r^2 \mu(r)}} = \frac{A_0}{r\sqrt{\mu(r)}}$$

### 2.4 Solution WKB complète (homogène)

$$\phi_{\text{hom}}(r) = \frac{A_0}{r\sqrt{\mu(r)}} \cos\left(\int^r \mu(r') dr' + \phi_0\right)$$

## 3. Solution particulière avec source

### 3.1 Méthode de la fonction de Green

La solution complète est :

$$\phi(r) = \phi_{\text{hom}}(r) + \phi_{\text{part}}(r)$$

où la solution particulière s'écrit via la fonction de Green :

$$\phi_{\text{part}}(r) = \int_0^\infty G(r, r') S(r') dr'$$

### 3.2 Fonction de Green WKB

Pour l'équation avec terme en 2/r, la fonction de Green WKB est :

$$G(r, r') = \frac{1}{r r' \sqrt{\mu(r)\mu(r')}} \times \frac{\sin|\theta(r) - \theta(r')|}{\mu_{\text{avg}}}$$

où $\mu_{\text{avg}}$ est une moyenne appropriée.

### 3.3 Simplification pour μ ≈ constant

Si μ varie lentement (ce qui est le cas loin du bord du halo), on peut approximer μ ≈ μ_0 = constant.

La solution particulière devient :

$$\phi_{\text{part}}(r) = -\frac{\alpha^*}{M_{\text{Pl}} \mu_0} \int_0^\infty \frac{\sin(\mu_0|r-r'|)}{r'} \rho_m(r') r'^2 dr'$$

## 4. Calcul de la densité d'énergie moyennée

### 4.1 Expression générale

La densité d'énergie du champ est :

$$\rho_\phi = \frac{1}{2}\left(\frac{d\phi}{dr}\right)^2 + \frac{1}{2}|\mu^2| \phi^2$$

### 4.2 Dérivée de la solution WKB

Pour $\phi = \frac{A_0}{r\sqrt{\mu}} \cos(\theta)$ :

$$\frac{d\phi}{dr} = -\frac{A_0}{r\sqrt{\mu}} \left[ \mu \sin\theta + \frac{1}{r}\cos\theta + \frac{\mu'}{2\mu}\cos\theta \right]$$

À l'ordre dominant (μr >> 1) :

$$\frac{d\phi}{dr} \approx -\frac{A_0 \mu}{r\sqrt{\mu}} \sin\theta = -\frac{A_0 \sqrt{\mu}}{r} \sin\theta$$

### 4.3 Moyennage sur une période

Sur une période d'oscillation T = 2π/μ :

$$\langle \sin^2\theta \rangle = \frac{1}{2}, \quad \langle \cos^2\theta \rangle = \frac{1}{2}, \quad \langle \sin\theta \cos\theta \rangle = 0$$

### 4.4 Termes moyennés

**Terme cinétique :**

$$\left\langle \left(\frac{d\phi}{dr}\right)^2 \right\rangle = \frac{A_0^2 \mu}{r^2} \langle \sin^2\theta \rangle = \frac{A_0^2 \mu}{2r^2}$$

**Terme de potentiel :**

$$\langle \mu^2 \phi^2 \rangle = \mu^2 \cdot \frac{A_0^2}{r^2 \mu} \langle \cos^2\theta \rangle = \frac{A_0^2 \mu}{2r^2}$$

### 4.5 Densité d'énergie moyennée

$$\boxed{\langle \rho_\phi \rangle = \frac{1}{2} \cdot \frac{A_0^2 \mu}{2r^2} + \frac{1}{2} \cdot \frac{A_0^2 \mu}{2r^2} = \frac{A_0^2 \mu}{2r^2}}$$

**Résultat crucial** : Si μ ≈ constant, alors $\langle \rho_\phi \rangle \propto 1/r^2$.

## 5. Détermination de l'amplitude A₀

### 5.1 À partir de la solution avec source

Pour une masse ponctuelle M à l'origine, la solution est (cf. dérivation précédente) :

$$\phi(r) = -\frac{\alpha^* M}{4\pi \mu M_{\text{Pl}}} \cdot \frac{\sin(\mu r)}{r}$$

Comparant avec la forme WKB $\phi = \frac{A_0}{r\sqrt{\mu}} \cos(\theta)$ :

$$A_0 = \frac{\alpha^* M}{4\pi \sqrt{\mu} M_{\text{Pl}}}$$

### 5.2 Densité d'énergie résultante

$$\langle \rho_\phi \rangle = \frac{1}{2r^2} \cdot \mu \cdot \frac{(\alpha^*)^2 M^2}{16\pi^2 \mu M_{\text{Pl}}^2}$$

$$\boxed{\langle \rho_\phi \rangle = \frac{(\alpha^*)^2 M^2}{32\pi^2 M_{\text{Pl}}^2} \cdot \frac{1}{r^2}}$$

## 6. Vérification dimensionnelle

En unités naturelles (ℏ = c = 1) :
- [α*] = 1 (sans dimension)
- [M] = [E] = [L⁻¹]
- [M_Pl] = [E] = [L⁻¹]
- [r] = [L]

$$[\rho_\phi] = \frac{[E]^2}{[E]^2} \cdot \frac{1}{[L]^2} = [L]^{-2}$$

Mais [ρ] devrait être [E⁴] = [L⁻⁴]. 

**Erreur identifiée !** Reprenons.

## 7. Correction : Analyse dimensionnelle rigoureuse

### 7.1 Dimensions correctes

En unités naturelles (ℏ = c = 1) :
- [φ] = [E] (le champ scalaire a dimension d'énergie)
- [∂φ/∂r] = [E]/[L] = [E²] (car [L] = [E⁻¹])
- [ρ_φ] = [(∂φ)²] = [E⁴] ✓

### 7.2 Reprise du calcul

La solution pour masse ponctuelle :

$$\phi(r) = -\frac{\alpha^* M}{4\pi \mu M_{\text{Pl}}} \cdot \frac{\sin(\mu r)}{r}$$

Vérifions les dimensions :
- [α*] = 1
- [M] = [E] (masse-énergie de la source)
- [μ] = [E] (masse du champ)
- [M_Pl] = [E]
- [r] = [E⁻¹]

$$[\phi] = \frac{[E]}{[E][E]} \cdot \frac{1}{[E^{-1}]} = \frac{[E]}{[E^2]} \cdot [E] = 1$$

**Problème** : φ devrait avoir dimension [E], pas être sans dimension.

### 7.3 Identification de l'erreur

Le terme source dans l'équation de KG est :

$$S = \frac{\alpha^*}{M_{\text{Pl}}} T^\mu_\mu$$

Pour de la matière non-relativiste :
$$T^\mu_\mu = -\rho c^2$$

où ρ est la densité de masse [M/L³] = [E⁴] en unités naturelles.

Donc :
$$[S] = \frac{1}{[E]} \cdot [E^4] = [E^3]$$

L'équation de KG :
$$\nabla^2 \phi - m^2 \phi = S$$

- $[\nabla^2 \phi] = [E^2] \cdot [E] = [E^3]$ ✓
- $[m^2 \phi] = [E^2] \cdot [E] = [E^3]$ ✓
- $[S] = [E^3]$ ✓

La solution via fonction de Green :
$$\phi \sim \frac{S}{\mu^2} \cdot \frac{1}{r}$$

$$[\phi] = \frac{[E^3]}{[E^2]} \cdot [E^{-1}] = [E] \cdot [E^{-1}] = 1$$

**Le problème persiste.** 

### 7.4 Solution : Formulation covariante

La fonction de Green du Laplacien 3D satisfait :
$$\nabla^2 G = \delta^3(\mathbf{r})$$

avec $[\delta^3] = [L^{-3}] = [E^3]$.

La solution est $G = -1/(4\pi r)$, donc $[G] = [E]$.

Pour l'équation $(\nabla^2 + \mu^2)\phi = S$ avec $[S] = [E^3]$ :

$$\phi(\mathbf{r}) = \int G(\mathbf{r}, \mathbf{r}') S(\mathbf{r}') d^3r'$$

$$[\phi] = [E] \cdot [E^3] \cdot [E^{-3}] = [E]$$ ✓

### 7.5 Reprise avec les bons facteurs

La fonction de Green de l'équation de Helmholtz :
$$(\nabla^2 + \mu^2) G = \delta^3(\mathbf{r})$$

est :
$$G(\mathbf{r}) = -\frac{e^{i\mu r}}{4\pi r}$$

Pour une source $S(\mathbf{r}') = (\alpha^*/M_{\text{Pl}}) \rho_m(\mathbf{r}')$ :

$$\phi(\mathbf{r}) = -\frac{\alpha^*}{4\pi M_{\text{Pl}}} \int \frac{e^{i\mu|\mathbf{r}-\mathbf{r}'|}}{|\mathbf{r}-\mathbf{r}'|} \rho_m(\mathbf{r}') d^3r'$$

Pour une masse ponctuelle $\rho_m = M \delta^3(\mathbf{r})$ :

$$\phi(r) = -\frac{\alpha^* M}{4\pi M_{\text{Pl}}} \cdot \frac{e^{i\mu r}}{r}$$

Partie réelle (solution physique) :

$$\phi(r) = -\frac{\alpha^* M}{4\pi M_{\text{Pl}}} \cdot \frac{\cos(\mu r)}{r}$$

ou avec sin selon les conditions aux limites.

## 8. Calcul correct de la densité d'énergie

### 8.1 Gradient

$$\frac{d\phi}{dr} = -\frac{\alpha^* M}{4\pi M_{\text{Pl}}} \cdot \frac{d}{dr}\left(\frac{\cos(\mu r)}{r}\right)$$

$$= -\frac{\alpha^* M}{4\pi M_{\text{Pl}}} \cdot \left(-\frac{\mu \sin(\mu r)}{r} - \frac{\cos(\mu r)}{r^2}\right)$$

$$= \frac{\alpha^* M}{4\pi M_{\text{Pl}}} \cdot \left(\frac{\mu \sin(\mu r)}{r} + \frac{\cos(\mu r)}{r^2}\right)$$

Pour μr >> 1, le terme dominant est :

$$\frac{d\phi}{dr} \approx \frac{\alpha^* M \mu}{4\pi M_{\text{Pl}}} \cdot \frac{\sin(\mu r)}{r}$$

### 8.2 Carré du gradient moyenné

$$\left\langle \left(\frac{d\phi}{dr}\right)^2 \right\rangle = \frac{(\alpha^*)^2 M^2 \mu^2}{16\pi^2 M_{\text{Pl}}^2} \cdot \frac{\langle\sin^2(\mu r)\rangle}{r^2}$$

$$= \frac{(\alpha^*)^2 M^2 \mu^2}{16\pi^2 M_{\text{Pl}}^2} \cdot \frac{1}{2r^2}$$

$$= \frac{(\alpha^*)^2 M^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2 r^2}$$

### 8.3 Terme de potentiel moyenné

$$\langle \mu^2 \phi^2 \rangle = \mu^2 \cdot \frac{(\alpha^*)^2 M^2}{16\pi^2 M_{\text{Pl}}^2} \cdot \frac{\langle\cos^2(\mu r)\rangle}{r^2}$$

$$= \frac{(\alpha^*)^2 M^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2 r^2}$$

### 8.4 Densité totale

$$\langle \rho_\phi \rangle = \frac{1}{2}\langle(\nabla\phi)^2\rangle + \frac{1}{2}\langle\mu^2\phi^2\rangle$$

$$= \frac{1}{2} \cdot \frac{(\alpha^*)^2 M^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2 r^2} + \frac{1}{2} \cdot \frac{(\alpha^*)^2 M^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2 r^2}$$

$$\boxed{\langle \rho_\phi \rangle = \frac{(\alpha^*)^2 M^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2 r^2}}$$

### 8.5 Vérification dimensionnelle finale

En unités naturelles :
- [α*] = 1
- [M] = [E]
- [μ] = [E]
- [M_Pl] = [E]
- [r] = [E⁻¹]

$$[\rho_\phi] = \frac{[E]^2 [E]^2}{[E]^2 [E]^{-2}} = \frac{[E]^4}{[E]^0} = [E^4]$$ ✓

## 9. Expression finale du profil

### 9.1 Forme générale

$$\boxed{\langle \rho_\phi(r) \rangle = \frac{(\alpha^*)^2 \mu^2}{32\pi^2 M_{\text{Pl}}^2} \cdot \frac{M^2}{r^2} \propto \frac{1}{r^2}}$$

### 9.2 En termes de paramètres HCM

Avec $\mu^2 = (\alpha^* M_{\text{Pl}})^2 [(\rho/\rho_c)^{2/3} - 1]$ :

Si ρ >> ρc (intérieur du halo), alors $\mu \approx \alpha^* M_{\text{Pl}} (\rho/\rho_c)^{1/3}$.

Pour un profil auto-cohérent où ρ ~ ρ_φ :

$$\rho_\phi \sim \frac{(\alpha^*)^2 (\alpha^* M_{\text{Pl}})^2 (\rho_\phi/\rho_c)^{2/3}}{M_{\text{Pl}}^2} \cdot \frac{M^2}{r^2}$$

$$\rho_\phi \sim (\alpha^*)^4 (\rho_\phi)^{2/3} \rho_c^{-2/3} \cdot \frac{M^2}{r^2}$$

$$\rho_\phi^{1/3} \sim (\alpha^*)^4 \rho_c^{-2/3} \cdot \frac{M^2}{r^2}$$

$$\boxed{\rho_\phi \propto \frac{1}{r^6}}$$

**Attention** : Ce résultat diffère du r⁻² obtenu pour μ = constant !

## 10. Réconciliation

### 10.1 Deux régimes

**Régime 1 : μ ≈ constant** (approximation)

Si on néglige la rétroaction de ρ_φ sur μ :
$$\langle \rho_\phi \rangle \propto \frac{1}{r^2}$$

C'est le cas si ρ_m >> ρ_φ (halo dominé par les baryons).

**Régime 2 : Auto-cohérent**

Si ρ_φ domine et détermine μ :
$$\langle \rho_\phi \rangle \propto \frac{1}{r^6}$$

### 10.2 Situation réaliste

Dans les galaxies réelles :
- Centre (r < quelques kpc) : baryons dominent → μ déterminé par ρ_m
- Halo (r > 10 kpc) : DM domine → régime auto-cohérent

Le profil réel est une **transition** entre ces deux régimes.

### 10.3 Profil effectif

La solution numérique du système couplé montrera probablement :
- Pente ~ -2 près du centre (régime forcé par les baryons)
- Pente plus raide (-3 à -6) à l'extérieur (régime auto-cohérent)
- Coupure à ρc (transition vers énergie noire)

Ceci ressemble qualitativement au profil NFW !

## 11. Conclusion

La dérivation WKB rigoureuse montre que :

1. **Profil isotherme (ρ ∝ r⁻²)** : valide quand μ est fixé par les baryons
2. **Profil plus raide (ρ ∝ r⁻⁶)** : valide en régime auto-cohérent
3. **Profil réaliste** : transition entre ces régimes ≈ NFW

La prochaine étape est de résoudre numériquement le système moyenné pour obtenir le profil exact.
