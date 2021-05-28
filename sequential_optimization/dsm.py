from pytikz.data_flow import DataFlow


data_flow = DataFlow()

data_flow.show_hierarchy_edges = True
data_flow.show_hierarchy_tree = True
data_flow.hierarchy_tree_side = 'right'
data_flow.hierarchy_tree_angle = 270
data_flow.hierarchy_tree_size_x = 1.
data_flow.default_dep_size_x = 1.5
data_flow.default_dep_size_y = 1.2

data_flow.add_comp('comp0', r'Optimizer', color=r'yellow!20')
data_flow.add_comp('comp1', r'Stiffness comp', color=r'blue!20')
data_flow.add_comp('comp2', r'K comp', color=r'blue!20')
data_flow.add_comp('comp3', r'Integrator', color=r'green!20')
data_flow.add_comp('comp4', r'kinematic comp', color=r'blue!20',size_x=3)
data_flow.add_comp('comp5', r'u comp', color=r'blue!20',size_x=3)
data_flow.add_comp('comp6', r'Integrator', color=r'green!20')
data_flow.add_comp('comp7', r'R comp', color=r'blue!20')
data_flow.add_comp('comp8', r'Integrator', color=r'green!20')
data_flow.add_comp('comp9', r'P comp', color=r'blue!20')
data_flow.add_comp('comp10', r'Backbone comp', color=r'blue!20',size_x=3)


# data_flow.add_comp('comp9', r'Stiffness comp', color=r'blue!20')
# data_flow.add_comp('comp10', r'K comp', color=r'blue!20')
# data_flow.add_comp('comp11', r'Integrator', color=r'green!20')
# data_flow.add_comp('comp12', r'kinematic comp', color=r'blue!20',size_x=3)
# data_flow.add_comp('comp13', r'u comp', color=r'blue!20',size_x=3)
# data_flow.add_comp('comp14', r'Integrator', color=r'green!20')
# data_flow.add_comp('comp15', r'R comp', color=r'blue!20')
# data_flow.add_comp('comp16', r'p comp', color=r'blue!20')

data_flow.add_dep('comp0', 'comp1', r'$D_{inner_{q,i}}, D_{outer_{q,i}}$',dep_size_x=3)
# data_flow.add_dep('comp0', 'comp9', r'$D_{inner_{q,i}}, D_{outer_{q,i}}$',dep_size_x=3)

data_flow.add_dep('comp0', 'comp2', r'$\kappa_{q,i},\beta_{q,i},L_{q,i}$',dep_size_x=3)
# data_flow.add_dep('comp0', 'comp10', r'$\kappa_{q,i}$')

data_flow.add_dep('comp0', 'comp4', r'$\alpha_{q,i},\beta_{q,i},L_{q,i}$',dep_size_x=3)
# data_flow.add_dep('comp0', 'comp12', r'$\alpha_{q,i},\beta_{q,i},L_{q,i},Lc_{q,i}$',dep_size_x=4)

data_flow.add_dep('comp1', 'comp2', r'$k_{q,ib},k_{q,it}$',dep_size_x=2)
# data_flow.add_dep('comp9', 'comp10', r'$k_{q,ib},k_{q,it}$',dep_size_x=2)
data_flow.add_dep('comp2', 'comp4', r'$K_{q,i}$')
data_flow.add_dep('comp3', 'comp4', r'$\psi_{q,i}$')
data_flow.add_dep('comp4', 'comp3', r'$\dot{\psi}_{q,i}$')
# data_flow.add_dep('comp10', 'comp12', r'$K_{q,i}$')
data_flow.add_dep('comp3', 'comp5', r'$\psi_{q,i}$')
data_flow.add_dep('comp5', 'comp7', r'$\hat{u}_{q,i}$')
# data_flow.add_dep('comp13', 'comp15', r'$\hat{u}_{q,i}$')
data_flow.add_dep('comp7', 'comp6', r'$\dot{R}$')
# data_flow.add_dep('comp15', 'comp14', r'$\dot{R}$')
data_flow.add_dep('comp6', 'comp9', r'$R_{q,i}$')
data_flow.add_dep('comp8', 'comp9', r'$P_{q,i}$')
data_flow.add_dep('comp9', 'comp8', r'$\dot{P}_{q,i}$')
data_flow.add_dep('comp9', 'comp10', r'$B_{q,i}$')
# data_flow.add_dep('comp14', 'comp16', r'$R$')
data_flow.add_dep('comp10', 'comp0', r'objective\\constraints',dep_size_x=2)
# data_flow.add_dep('comp11', 'comp12', r'$\psi_{q,i}$')
# data_flow.add_dep('comp12', 'comp11', r'$\dot{\psi}_{q,i}$')
# data_flow.add_dep('comp11', 'comp13', r'$\psi_{q,i}$')
# data_flow.add_dep('comp16', 'comp0', r'objective\\constraints',dep_size_x=2)

data_flow.add_group('group1', 'Kinematic', ('comp1', 'comp2','comp3','comp4'))
data_flow.add_group('group2', 'Backbone', ('comp5','comp6','comp7','comp8','comp9','comp10'))

# data_flow.add_group('group1', 'Integrator group', ('comp3', 'comp4'))
# data_flow.add_group('group2', 'Integrator group', ('comp8', 'comp9'))
# data_flow.add_group('group3', 'Model', ('comp1', 'comp2','comp5','comp10'))
# data_flow.add_group('group1', 'Right-arm', ('comp1', 'comp2','comp3','comp4','comp5','comp6','comp7','comp8'))
# data_flow.add_group('group2', 'Left-arm', ('comp9','comp10', 'comp11','comp12','comp13','comp14','comp15','comp16'))

data_flow.write('data_flow')