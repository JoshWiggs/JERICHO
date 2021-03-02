"""
Module containing set of functions utilisised to perform different forms of numerical different

 .. moduleauthor:: J. A. Wiggs <j.wiggs@lancaster.ac.uk>

Dependencies
------------
* Numpy_: From the python package index, a fundamental package for scientific computing with Python.

.. _Numpy: https://numpy.org/
"""

import numpy as np

class diff:
    """
    These differential functions assume that the arrays they are being applied to have a layer of ghost cells surrounding them and that the dimensions are structured in the order one would expect for a Euclidean space, that is (x,y,z). These functions differentiate using the well-known center-differencing method as it is proven to increase stability.
    """
    def x(array,dx,Boundary_Type='None'):
        """
        Solver for the x-dimension

        **Arguments**

        array
            Array that contains the variable that is to be differentiated

        dx
            Size of grid spacing in the x-direction

        **Keywords**

        Boundary_Type
            Default None, but can be set to Periodic in order to wrap the ghost cells around so they connect and communicate with one another
        """
        dimension_number = int(len(np.shape(array)))
        output = np.zeros_like(array)
        try:
            if dimension_number == 1:
                output[1:-1] = ((array[2:] - array[:-2]) / (2*dx))
                return output
            elif dimension_number == 2:
                output[:,1:-1] = ((array[:,2:] - array[:,:-2]) / (2*dx))
                if Boundary_Type == 'Periodic':
                    output[:,0] = ((array[:,1]-array[:,-1]) / (2*dx))
                    output[:,-1] = ((array[:,0]-array[:,-2]) / (2*dx))
                return output
            elif dimension_number == 3:
                output[:,:,1:-1] = ((array[:,:,2:]-array[:,:,:-2]) / (2*dx))
                return output
            elif dimension_number > 3:
                raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
        except:
            return output
    def y(array,dy,Boundary_Type='None'):
        """
        Solver for the y-dimension

        **Arguments**

        array
            Array that contains the variable that is to be differentiated

        dy
            Size of grid spacing in the y-direction

        **Keywords**

        Boundary_Type
            Default None, but can be set to Periodic in order to wrap the ghost cells around so they connect and communicate with one another
        """
        dimension_number = int(len(np.shape(array)))
        output = np.zeros_like(array)
        try:
            if dimension_number == 1:
                return output
            if dimension_number == 2:
                output[1:-1,:] = ((array[2:,:] - array[:-2,:]) / (2*dy))
                if Boundary_Type == 'Periodic':
                    output[0,:] = ((array[1,:]-array[-1,:]) / (2*dy))
                    output[-1,:] = ((array[0,:]-array[-2,:]) / (2*dy))
                return output
            elif dimension_number == 3:
                output[:,1:-1,:] = ((array[:,2:,:]-array[:,:-2,:]) / (2*dy))
                return output
            elif dimension_number > 3:
                raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
        except:
            return output
    def z(array,dz):
        """
        Solver for the z-dimension

        **Arguments**

        array
            Array that contains the variable that is to be differentiated

        dz
            Size of grid spacing in the z-direction
        """
        dimension_number = int(len(np.shape(array)))
        output = np.zeros_like(array)
        try:
            if dimension_number == 1:
                return output
            if dimension_number == 2:
                return output
            if dimension_number == 3:
                output[1:-1,:,:] = ((array[2:,:,:]-array[:-2,:,:]) / (2*dz))
                return output
            elif dimension_number > 3:
                raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
        except:
            return output
    class euler_forward:
        """
        These differential functions assume that the arrays they are being applied to have a layer of ghost cells surrounding them and that the dimensions are structured in the order one would expect for a Euclidean space, that is (x,y,z). These functions differentiate using Euler forward differencing and are specially included for use in the magnetic field solver.
        """
        def x(array,dx):
            """
            Solver for the x-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dx
                Size of grid spacing in the x-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    output[1:-1] = ((array[2:] - array[1:-1]) / (dx))
                    return output
                elif dimension_number == 2:
                    output[:,1:-1] = ((array[:,2:] - array[:,1:-1]) / (dx))
                    return output
                elif dimension_number == 3:
                    output[:,:,1:-1] = ((array[:,:,2:]-array[:,:,1:-1]) / (dx))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
        def y(array,dy):
            """
            Solver for the y-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dy
                Size of grid spacing in the y-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    return output
                if dimension_number == 2:
                    output[1:-1,:] = ((array[2:,:] - array[1:-1,:]) / (dy))
                    return output
                elif dimension_number == 3:
                    output[:,1:-1,:] = ((array[:,2:,:]-array[:,1:-1,:]) / (dy))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
        def z(array,dz):
            """
            Solver for the z-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dz
                Size of grid spacing in the z-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    return output
                if dimension_number == 2:
                    return output
                if dimension_number == 3:
                    output[1:-1,:,:] = ((array[2:,:,:]-array[1:-1,:,:]) / (dz))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
    class euler_backward:
        """
        These differential functions assume that the arrays they are being applied to have a layer of ghost cells surrounding them and that the dimensions are structured in the order one would expect for a Euclidean space, that is (x,y,z). These functions differentiate using Euler backwards differencing and are specially included for use in the magnetic field solver.
        """
        def x(array,dx):
            """
            Solver for the x-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dx
                Size of grid spacing in the x-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    output[1:-1] = ((array[1:-1] - array[:-2]) / (dx))
                    return output
                elif dimension_number == 2:
                    output[:,1:-1] = ((array[:,1:-1] - array[:,:-2]) / (dx))
                    return output
                elif dimension_number == 3:
                    output[:,:,1:-1] = ((array[:,:,1:-1]-array[:,:,:-2]) / (dx))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
        def y(array,dy):
            """
            Solver for the y-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dy
                Size of grid spacing in the y-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    return output
                if dimension_number == 2:
                    output[1:-1,:] = ((array[1:-1,:] - array[:-2,:]) / (dy))
                    return output
                elif dimension_number == 3:
                    output[:,1:-1,:] = ((array[:,1:-1,:]-array[:,:-2,:]) / (dy))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
        def z(array,dz):
            """
            Solver for the z-dimension

            **Arguments**

            array
                Array that contains the variable that is to be differentiated

            dz
                Size of grid spacing in the z-direction
            """
            dimension_number = int(len(np.shape(array)))
            output = np.zeros_like(array)
            try:
                if dimension_number == 1:
                    return output
                if dimension_number == 2:
                    return output
                if dimension_number == 3:
                    output[1:-1,:,:] = ((array[1:-1,:,:]-array[:-2,:,:]) / (dz))
                    return output
                elif dimension_number > 3:
                    raise Exception('This function only supports up to 3 dimensional arrays, the given input array contains {} dimensions'.format(dimension_number))
            except:
                return output
