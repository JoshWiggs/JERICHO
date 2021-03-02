================================================
JERICHO-PY's Documentation
================================================

**JERICHO-PY** (Jovian magnEtospheRIC Ion kinetic fluid electron Hybrid plasma mOdel in PYthon) is a hybrid kinetic-ion, fluid electron hybrid plasma model implemented using Python 3. Developed for the examination of magnetospheric plasma, specifically in the Jovian system. Ions are described kinetically, with their equation of motion including centrifugal forces, electrons are formed into a single fluid continuum, described using the MHD equations.

This version was developed to allow for the quick prototyping and implementation of different physical effects in order to determine the size of their impact on the global domain. Additional, it is hoped that version will have pedagogical value to both students and scientists new to hybrid plasma modelling.

Dependencies
############
* Python 3.7 (required)
* Numpy 1.18.1 (required)
* matplotlib 3.2.0 (required for plotting operations)
* h5py 2.10.0 (needed for HDF5 IO method)
* scipy 1.4.1 (needed for HDF5 IO method)

*All versions are minimums, any version after should also work*

Publications
#############
* Wiggs, J. A. & Arridge, C. S., 2021. *JERICHO-PY: A Hybrid Ion-Kinetic, Fluid-Electron Plasma Model for the Outer Planets in Python* (in prep)

C++
####
JERICHO has been ported into c++ and this is the version of the code currently being used for production runs examining the Jovian magnetosphere. This codebase is not currently available for public use, however we are open to collaboration so feel free to get in touch.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Home <self>
   classes
   functions
   numerical_differentials
   io
   support

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
