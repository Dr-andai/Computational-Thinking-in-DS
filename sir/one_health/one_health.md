# SIR MODEL
**Hello world of epidemic modeling**
The SIR model is based on the idea that a population during an outbreak can be divided into three groups or compartments: S – susceptible, I – infected, and R – recovered. The model was first described by Kermack and McKendrick. Differential equations describe the SIR model mathematically. The equations represent the rate of change of each compartment over time rather than the absolute number of individuals in a compartment at any given time.
- dS/dt = -𝛽SI/N # (𝛽 i sthe transmission rate)
- dI/dt = 𝛽SI/N - YI # (Y is the recovery rate)
- dR/dt = YI

The rate of overall infection is dependent on the number of susceptible individuals and infected individuals at a given time, as well as the fixed transmission rate parameter, 𝛽. When you plug in all the numbers, the result of each equation gives units in number of people per time step.

# One Health
One Health is the concept that human, animal and environmental health are continuous, and should be addressed continuously.
Let's take an example, rift valley fever. Rift Valley Fever (RVF) is a viral disease transmitted by mosquitoes that mainly affects livestock. It can also infect humans. While most human cases remain mild, it can cause death.

*Reference: GAVI* [GAVI](https://www.gavi.org/vaccineswork/rift-valley-fever-what-it-how-it-spreads-and-how-stop-it)
**How it’s transmitted**
In animals, the disease is mainly spread through bites from infected mosquitoes. At least 50 mosquito species can transmit the Rift Valley fever virus, including Aedes, Culex, Anopheles and Mansonia species. Mosquitoes become infected when they feed on animals carrying the virus in their blood, then transmit it to other animals through their bites. In Aedes mosquitoes, vertical transmission – from infected females to their eggs – is also possible, allowing the virus to survive in the environment.

For humans, the most common way to get infected is through direct contact with the blood or organs of an infected animal. This often happens during veterinary work, slaughtering, or butchering.

While it is also possible for human to get the virus from a mosquito bite, this is not common. No human-to-human transmission has been observed to date.


# ONE HEALTH SIR
The traditional SIR works well for human specific diseases. Example smallpox/measeles. It has also been used in COVID 19 pandemic. Where after the disease commenced, the virus was transmitted via human-human, and did not need any vector. Traditional SIR falls short because it lacks:  
- animal reservoir (the pathogen should persist on the organism)  
- Cross spices spill over  
- Environment Persistence

# Theoretical framework for ONEHEALTH SIR MODEL  
- Incorporating multiple species and environmental compartments
- Critical transmission pathways: Livestock -> Mosquitoes -> Humans, Environment -> Mosquitoes breeding
- Seasonal effects: Rainfall patterns which create mosquito breeding habitats leading to outbreak cycles
