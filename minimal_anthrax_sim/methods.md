# Methods: Environmental Pathogen Transmission Model

## Model Structure
We developed a deterministic compartmental model to simulate the transmission of a persistent environmental pathogen in a system consisting of livestock hosts, an environmental reservoir, and scavengers. The model tracks four state variables: Susceptible Hosts ($S$), Infected Hosts ($I$), Infectious Carcasses ($C$), and Environmental Pathogen Reservoir ($E$). This framework builds upon established epidemiological models where host mortality generates infectious patches in the environment, as seen in models of anthrax transmission [[1]](https://doi.org/10.1007/s11538-016-0238-1).

## Host and Pathogen Dynamics
Host population dynamics follow a logistic growth function with carrying capacity $K_{cap}$ and natural mortality $\mu$. Susceptible individuals ($S$) contract the pathogen through contact with the environmental reservoir ($E$) at a transmission rate $\beta$. Infected individuals ($I$) succumb to the disease at rate $\alpha$ (the sum of disease-induced and natural mortality), transitioning into the Carcass state ($C$). The disease-induced mortality rate was parameterized to correspond to an average infectious period of approximately 7 days [[1]](https://doi.org/10.1007/s11538-016-0238-1).

## The Scavenger Filter Mechanism
The core of this model investigates the role of scavengers in altering the reproductive number $R_0$ by removing infectious biomass before it can contaminate the environment. Carcasses ($C$) have two competing fates:

1.  **Reservoir Formation:** Carcasses decay naturally at rate $\kappa$, releasing a burst of pathogen load $\phi$ into the environment to form a localized infectious zone (LIZ). The decay rate $\kappa$ was set to $0.1$ day$^{-1}$, representing an average decay period of 10 days in the absence of scavenging [[1]](https://doi.org/10.1007/s11538-016-0238-1).
2.  **Scavenging:** Scavengers consume carcasses at rate $\sigma$. We assume scavenged carcasses are effectively removed from the transmission cycle without contributing to the environmental reservoir.

## Reproductive Number and Scavenger Control
The reproductive number component for hosts ($K_{LL}$) is proportional to the probability that a carcass remains in the environment to form a LIZ. In our ODE, this probability is governed by the competition between natural decay ($\kappa$) and scavenging ($\sigma$):

$$
(1 - \gamma_L) = \frac{\kappa}{\kappa + \sigma}
$$

Where $\gamma_L$ represents the "scavenging capacity" or the proportion of carcasses removed:
* **Low Scavenging ($\sigma = 0$):** $(1 - \gamma_L) = 1$. All infected carcasses degrade naturally and contaminate the environment.
* **High Scavenging ($\sigma \gg \kappa$):** $(1 - \gamma_L) \to 0$. Most carcasses are removed before they can establish an environmental reservoir, effectively driving $R_0 < 1$.

## Parameterization
Parameters were derived from literature on environmental pathogen persistence and host dynamics:
* **Carcass decay ($\kappa$):** Set to $0.1$ day$^{-1}$ (approx. 10 days) [[1]](https://doi.org/10.1007/s11538-016-0238-1).
* **Disease mortality ($\alpha$):** Set to $0.14$ day$^{-1}$ (approx. 7 days) [[1]](https://doi.org/10.1007/s11538-016-0238-1).
* **Host Lifespan ($\mu$):** Set to correspond to an average lifespan of 5 years ($1/\mu \approx 2000$ days), consistent with livestock demographics in similar systems [[2]](https://doi.org/10.1038/s41598-020-72440-6).
* **Pathogen persistence ($\delta$):** Assumed to be long-term with a low decay rate, as spores or infectious agents can persist in the environment for years to decades [[2]](https://doi.org/10.1038/s41598-020-72440-6) [[3]](https://doi.org/10.1098/rspb.2023.2568).

### References
[1] Saad-Roy, C. M., van den Driessche, P., & Yakubu, A. A. (2017). A mathematical model of anthrax transmission in animal populations. *Bulletin of Mathematical Biology*, 79, 303-324. [https://doi.org/10.1007/s11538-016-0238-1](https://doi.org/10.1007/s11538-016-0238-1)

[2] Stella, E., Mari, L., Gabrieli, J., Barbante, C., & Bertuzzo, E. (2020). Permafrost dynamics and the risk of anthrax transmission: a modelling study. *Scientific Reports*, 10(1), 16460. [https://doi.org/10.1038/s41598-020-72440-6](https://doi.org/10.1038/s41598-020-72440-6)

[3] Dolfi, A. C., Kausrud, K., Rysava, K., Champagne, C., Huang, Y. H., Barandongo, Z. R., ... & Turner, W. C. (2024). Season of death, pathogen persistence and wildlife behaviour alter number of anthrax secondary infections from environmental reservoirs. *Proceedings of the Royal Society B*, 291(2014), 20232568. [https://doi.org/10.1098/rspb.2023.2568](https://doi.org/10.1098/rspb.2023.2568)