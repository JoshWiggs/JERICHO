JERICHO - Hybrid Plasma Model
=============================

**Authors: [Josh
Wiggs](https://www.lancaster.ac.uk/physics/about-us/people/josh-wiggs) &
[Chris
Arridge](https://www.lancaster.ac.uk/physics/about-us/people/chris-arridge)**

JERICHO (**J**ovian magn**E**tosphe**RIC** **I**on kinetic fluid
electron **H**ybrid plasma m**O**del) is a hybrid plasma model developed
for the examination of magnetospheric plasma, specifically in the Jovian
system. Ions are described kinetically, with their equation of motion
including centrifugal forces, electrons are formed into a single fluid
continuum, described using the MHD equations.

Python
------

JERICHO-PY is the python version of the model codebase. This version was
developed to allow for the quick prototyping and implementation of
different physical effects in order to determine the size of their
impact on the global domain. Additional, it is hoped that version will
have pedagogical value to both students and scientists new to hybrid
plasma modelling.

### Dependencies

-   Python 3.7 (required)
-   Numpy 1.18.1 (required)
-   matplotlib 3.2.0 (needed for plotting operations)
-   h5py 2.10.0 (needed for HDF5 IO method)
-   scipy 1.4.1 (needed for HDF5 IO method)

*All versions are minimums, any version after should also work*

### Future Work

-   Generalisation of model topology from 2.5D to 3D
-   Parallelisation of serial code for performance improvements
-   Improvement of numerical solvers to higher orders to allow
    resolution of more complex physics

Installation & Usage
--------------------

No special installation process is required for the deployment of
JERICHO-PY. Including the complete contents of this directory on a
machine with Python 3.7 or later and the required dependencies (see
above for these) available is sufficient, for none default options
additional dependencies may be required (again, see the list above). The
model can then be executed by running `2_5d_hybrid_model.py`. Input
parameters utilised by JERICHO-PY are stored separately in the
`parameters.py` file and can be edited directly. If an output IO method
is selected then these files will be generated inside the root of this
directory.

More complete documentation is located in the docs sub-directory.
Documentation is constructed using Sphinx 2.3 and can be rebuilt by
using the Make file located within this sub-directory. The standard
documentation provided with JERICHO is built using `make html`.

### Publications

-   Wiggs, J. A. & Arridge, C. S., 2021. *JERICHO-PY: A Hybrid
    Ion-Kinetic, Fluid-Electron Plasma Model for the Outer Planets in
    Python* (in prep)

C++
---

JERICHO has been ported into c++ and this is the version of the code
currently being used for production runs examining the Jovian
magnetosphere. This codebase is not currently available for public use,
however we are open to collaboration so feel free to get in touch.

Directory Structure
-------------------

-   Python: *Directory containing the JERICHO-PY program*
-   2_5d_hybrid_model.py: *Main program file for JERICHO-PY contains
    model setup, loop and IO*
-   parameters.py: *Contains the controls for options and definition of
    the variables utilised in the main program file for model runs*
-   docs: *Directory containing program documentation*
    -   Makefile: *Basic makefile for building Sphinx documentation*
    -   build: *Directory containing pre-built documentation for
        JERICHO-PY*
    -   html: *Directory containing HTML version of the documentation*
        -   classes.html: *Page containing documentation for the classes
            module*
        -   functions.html: *Page containing documentation for the
            functions module*
        -   index.html: *Index page of documentation*
        -   io.html: *Page containing documentation for the IO module*
        -   numerical_differentials.html: *Page containing
            documentation for the numerical_differentials module*
        -   support.html: *Page containing information for obtaining
            support using JERICHO.py*
    -   Source: *Directory containing restructured text file for
        generating documentation*
    -   classes.rst: *File for generating a page containing
        documentation for the classes module*
    -   functions.rst: *File for generating a page containing
        documentation for the functions module*
    -   index.rst: *File for generating the index page of the
        documentation*
    -   io.rst: *File for generating a page containing documentation for
        the IO module*
    -   numerical_differentials.rst: *File for generating a page
        containing documentation for the numerical_differentials
        module*
    -   support.rst: *File for generating a page containing information
        for obtaining support using JERICHO.py*
-   Modules: *Directory containing functions used by JERICHO-PY*
    -   __init__.py: *Identifier for Python package directory*
    -   classes.py: *Module containing all custom classes used in
        JERICHO.PY*
    -   functions.py: *Module containing general custom functions used
        in JERICHO.PY*
    -   io.py: *Module containing procedures for using HDF5 IO methods*
    -   numerical_differentials.py: *Module containing functions
        specifically for performing numerical differentiation*
-   .gitingore: *Standard Python .gitingore*
-   LICENSE.md: *GPL3 license*
-   README.md: *Contains general information about JERICHO-PY along with
    basic instructions for installation and usage*
-   README.txt: *Contains general information about JERICHO-PY along
    with basic instructions for installation and usage*


