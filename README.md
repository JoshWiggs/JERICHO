# JERICHO - Hybrid Plasma Model
**Authors: [Josh Wiggs](https://www.lancaster.ac.uk/physics/about-us/people/josh-wiggs) & [Chris Arridge](https://www.lancaster.ac.uk/physics/about-us/people/chris-arridge)**

JERICHO (**J**ovian magn**E**tosphe**RIC** **I**on kinetic fluid electron **H**ybrid plasma m**O**del) is a hybrid plasma model developed for the examination of magnetospheric plasma, specifically in the Jovian system. Ions are described kinetically, with their equation of motion including centrifugal forces, electrons are formed into a single fluid continuum, described using the MHD equations.

## Python
JERICHO-PY is the python version of the model codebase. This version was developed to allow for the quick prototyping and implementation of different physical effects in order to determine the size of their impact on the global domain. Additional, it is hoped that version will have pedagogical value to both students and scientists new to hybrid plasma modelling.  

### Dependencies
- Python 3.7 (required)
- Numpy 1.18.1 (required)
- matplotlib 3.2.0 (needed for plotting operations)
- h5py 2.10.0 (needed for HDF5 IO method)
- scipy 1.4.1 (needed for HDF5 IO method)

*All versions are minimums, any version after should also work*

### Future Work
- Generalisation of model topology from 2.5D to 3D
- Parallelisation of serial code for performance improvements
- Improvement of numerical solvers to higher orders to allow resolution of more complex physics

### Publications
- Wiggs, J. A. & Arridge, C. S., 2021. *JERICHO-PY: A Hybrid Ion-Kinetic, Fluid-Electron Plasma Model for the Outer Planets in Python* (in prep)

## C++
JERICHO has been ported into c++ and this is the version of the code currently being used for production runs examining the Jovian magnetosphere. This codebase is not currently available for public use, however we are open to collaboration so feel free to get in touch.
