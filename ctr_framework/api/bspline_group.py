import numpy as np
from openmdao.api import group



class BsplineGroup(group):
    '''
    BsplineGroup is a OpenMDAO Group object that includes all the components forming the optimization problem for finding a collision
    free path for the robot to follow. BsplineGroup uses the 3D B-spline function and a continuous function of anatomical constraints
    to ensure the B-spline curve is inside the anatomy. 
    
    '''
    def initialize(self):
        self.options.declare('filename')
        self.options.declare('r2')
        self.options.declare('r1')
        self.options.declare('sp')
        self.options.declare('fp')
        self.options.declare('num_cp', default=25, types=int)
        self.options.declare('num_pt', default=100, types=int)
        
        

    def setup(self):
        filename = self.options['filename']
        r2 = self.options['r2']
        r1 = self.options['r1']
        sp = self.options['sp']
        fp = self.options['fp']
        num_cp = self.options['num_cp']
        num_pt = self.options['num_pt']
        
        