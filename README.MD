# Triplet experiment Simulation

This project simulates the kinematics and resulting relativistic effects of three observers (A, B, and C) traveling through the solar system. By comparing their speeds and trajectories to theoretical "virtual objects" in perfect, counter-clockwise circular orbits, this simulation models the kinematic time dilation each observer experiences.

## The Local Inertial Frame (LIF)
In this model, the "virtual objects" act as proxies for the local inertial frame in free fall at any given distance from the Sun. This concept is directly analogous to how the Earth-Centered Inertial (ECI) frame is used in real-world physics; for instance, the kinematic time dilation of GPS atomic clocks is calculated based on their velocity relative to the ECI.

## Kinematic Time Dilation Profiles
Because time dilation is dependent on the observer's velocity relative to their local inertial frame, the three observers experience significantly different relativistic effects throughout their journeys:
* **Observer A (Earth Baseline):** Remains in a perfect circular orbit, matching the velocity of the local virtual object. Consequently, Observer A experiences no relative kinematic time dilation and serves as our baseline clock.
* **Observer B:** Departs at $t=0$ with a trajectory that creates a high initial velocity relative to the local frame (equivalent to 90 km/s). This results in a high initial rate of time dilation. As Observer B travels further from the Sun, this time dilation gradually decreases, asymptotically settling at a rate corresponding to their constant 60 km/s travel speed..
* **Observer C:** Departs half a year later from the opposite side of the Sun. Due to Earth's orbital direction at that specific time, Observer C's initial velocity relative to the local frame is much lower (equivalent to 30 km/s), resulting in significantly less initial time dilation. Like B, C's time dilation also gradually evolves toward the 60 km/s baseline as they move out of the inner solar system.
* **The Convergence** After approximately 1,000 days, observers B and C reach identical velocity vectors relative to the local rest frame, maintaining a fixed distance from one another and experiencing the exact same rate of time dilation.

## Invariance of the Local Speed of Light ($c$)
A fundamental premise of this model is that light propagates at speed $c$ relative to the Local Inertial Frame (the co-rotating "virtual object" at any given location). However, this does not mean that Observers B and C measure a different speed of light as they cut across these moving layers of space. Because an observer's motion relative to the Local Inertial Frame ($\vec{v}_{relative} = \vec{v}_{observer} - \vec{v}_{LIF}$) directly governs their physical measuring instruments, three interconnected relativistic effects continuously adapt to their instantaneous relative velocity:
* **Time Dilation:** The moving observer's onboard clock ticks slower by the Lorentz factor $\gamma = (1 - v_{relative}^2/c^2)^{-1/2}$.
* **Length Contraction:** Physical rulers and spatial intervals along the direction of relative motion contract by the same factor $\gamma$.
* **Relativity of Simultaneity:** Onboard clocks synchronized by the observer along their travel axis naturally acquire a phase offset ($\Delta t = v_{relative} L / c^2$) relative to the local background frame.

Together, these local Lorentz transformations dynamically recalibrate each observer's units of time and distance at every step of their journey. Whether an observer is co-moving with the frame like Observer A ($v_{relative} = 0$), launching "upstream" at $90\text{ km/s}$ like Observer B, or launching "downstream" at $30\text{ km/s}$ like Observer C, the physical adjustment of their clocks and rulers guarantees that all three observers always measure the local speed of light to be exactly $c$ within their own frame of reference.

## The Physics & Simulation Parameters

The simulation is set up with the following parameters:
* **The Sun** is stationary at the origin `(x=0, y=0)`.
* **Observer A (Earth)** starts at `x=0, y=1 AU` and travels in a counter-clockwise circular orbit.
* **Observer B** departs Earth at `t=0` and travels in a straight line along the x-axis at a constant speed of 60 km/s.
* **Observer C** remains on Earth for half a year (until Earth reaches `x=0, y=-1 AU`), then departs in a straight line along the x-axis at a constant speed of 60 km/s.

### The Virtual Object Calculation
For any point `(x, y)` in space, an object is at a distance `r = √(x² + y²)` from the Sun. The orbital velocity for a perfect circular orbit at that distance is calculated as:

$$v_{circ} = \sqrt{\frac{GM_{Sun}}{r}}$$

Because the orbit is counter-clockwise, the direction of the velocity vector is tangent to the circle. The normalized tangent vector for a point `(x, y)` is `(-y/r, x/r)`. The velocity vector of the virtual object is therefore:

$$\vec{v}_{virtual} = v_{circ} \cdot \left(-\frac{y}{r}, \frac{x}{r}\right)$$

At every daily time step, the script calculates the vector difference in velocity between each observer and this theoretical virtual object.

## Activate the environment

* conda env create -f environment.yml
* conda activate triplet-experiment

## Run the simulation

python main.py

This will create the file: 
* velocity_difference_plot.png
![Velocity Difference Plot](velocity_difference_plot.png)

python main_animated.py

This will create the file: 
* triplet_animation.mp4
<video src="[triplet_animation.mp4](https://emea01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fphotos.app.goo.gl%2F37qwDgAj4XBekf6PA&data=05%7C02%7C%7C4a428c70ab1c4dbc010208df32a2c935%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C639285346874380282%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=MM%2FmXy6r3qgI5f34kQcaHDwWlPqrHGdyR06O1BN7SM0%3D&reserved=0)" width="100%" controls autoplay loop muted playsinline>
  Your browser does not support the video tag.
</video>

The video animation visually maps the trajectories of the three observers over a 1,000-day period. It features a dynamic spatial map demonstrating the physical vector subtraction 

$$\vec{v}_{relative} = \vec{v}_{observer} - \vec{v}_{LIF}$$

alongside a real-time graph of the resulting clock rate differences as Observers B and C travel through the solar system.
