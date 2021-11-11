User defined objectives
=================================

Optimization problem
--------------------
In this tutorial, we will show you how to construct and use the sequential optimization problem in OpenMDAO.

The CTR sequential optimization group is as follows:

.. code-block:: python
        
            '''objectives'''
            desiredpointscomp = DesiredpointsComp(num_nodes=num_nodes,k=k)
            self.add_subsystem('Desiredpointscomp', desiredpointscomp, promotes=['*'])
            reachtargetptscomp = ReachtargetptsComp(k=k,targets = pt)
            self.add_subsystem('reachtargetptsComp', reachtargetptscomp, promotes=['*'])
            targetnormcomp = TargetnormComp(k=k)
            self.add_subsystem('Targetnormcomp', targetnormcomp, promotes=['*'])
            dpcomp = DpComp(k=k,num_nodes=num_nodes,p_=p_)
            self.add_subsystem('DpComp', dpcomp, promotes=['*'])
            crosssectioncomp = CrosssectionComp(k=k,num_nodes=num_nodes)
            self.add_subsystem('CrosssectionComp', crosssectioncomp, promotes=['*'])
            signedfuncomp = SignedfunComp(k=k,num_nodes=num_nodes,normals=normals)
            self.add_subsystem('SignedfunComp', signedfuncomp, promotes=['*'])
            equdply = EqudplyComp(k=k,num_nodes=num_nodes)
            self.add_subsystem('EqudplyComp', equdply, promotes=['*'])

            # objective function
            dl0 = init_guess['tube_section_length'] + init_guess['beta']
            norm1 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:],ord=1.125)
            norm2 = (dl0[:,0] - dl0[:,1])**2 + (dl0[:,1] -  dl0[:,2])**2
            norm3 = np.linalg.norm(pt_full[0,:]-pt_full[-1,:])/viapts_nbr
            norm4 = 2
            norm5 = 2*np.pi 

            objscomp = ObjsComp(k=k,num_nodes=num_nodes,
                                zeta=zeta,
                                    rho=rho,
                                        eps_r=eps_r,
                                            eps_p=eps_p,
                                                lag=lag,
                                                    norm1 = norm1,
                                                        norm2 = norm2,
                                                            norm3 = norm3,
                                                                norm4 = norm4,
                                                                    norm5 = norm5,
                                                                        eps_e = eps_e,)                                    
            self.add_subsystem('ObjsComp', objscomp, promotes=['*'])
            self.add_objective('objs')
        
        


.. toctree::
  :maxdepth: 2
  :titlesonly:
