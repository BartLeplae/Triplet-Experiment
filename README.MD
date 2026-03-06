# Triplet experiment Simulation

This project simulates the velocities of three observers (A, B, and C) traveling through the solar system. It compares their speeds and trajectories to theoretical "virtual objects" in perfect counter-clockwise circular orbits around the Sun at their respective distances. The time dilation experienced by the 3 observers is a function of this velocities.

* **Observer A (Earth)** no time dilation.
* **Observer B** time dilation corresponding with 90 km/s at take-off and gradually evolves to a time dilation corresponding with 60 km/s.
* **Observer C** time dilation corresponding with 30 km/s at take-off and gradually evolves to a time dilation corresponding with 60 km/s.
* After 1000 days, Observer B and C experience the same time dilation which corresponds with the fact that they keep the same distance with each other

## 🪐 The Physics & Simulation Parameters

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

## ctivate the environment

* conda env create -f environment.yml
* conda activate triplet-experiment

## Run the simulation

python main.py

This will create the file: 
* velocity_difference_plot.png
