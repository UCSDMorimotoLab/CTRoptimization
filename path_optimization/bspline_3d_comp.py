import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent


def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)

    knots = np.zeros(num_cp + order)
    knots[order-1:num_cp+1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0

    t_vec = np.linspace(0, 1, num_pt)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_pt, num_cp),
    )


class BsplineComp(ExplicitComponent):
    """
    General function to translate from control points to actual points
    using a b-spline representation.
    """

    def initialize(self):
        self.options.declare('num_pt', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('jac')
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        num_pt = self.options['num_pt']
        num_cp = self.options['num_cp']
        jac = self.options['jac']
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        self.add_input(in_name, shape=(num_cp, 3))
        self.add_output(out_name, shape=(num_pt, 3))

        jac = self.options['jac'].tocoo()
        nnz = len(jac.data)

        data = np.zeros((nnz, 3))
        rows = np.zeros((nnz, 3), int)
        cols = np.zeros((nnz, 3), int)

        for ind in range(3):
            data[:, ind] = jac.data
            rows[:, ind] = 3 * jac.row + ind
            cols[:, ind] = 3 * jac.col + ind

        data = data.flatten()
        rows = rows.flatten()
        cols = cols.flatten()

        self.declare_partials(out_name, in_name, val=data, rows=rows, cols=cols)

        self.full_jac = scipy.sparse.csc_matrix(
            (data, (rows, cols)), 
            shape=(3 * num_pt, 3 * num_cp),
        )

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        jac = self.options['jac']

        outputs[out_name] = jac.dot(inputs[in_name])


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    np.random.seed(0)

    num_cp = 300
    num_pt = 1500
    jac = get_bspline_mtx(num_cp, num_pt)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('cp', val=np.random.rand(num_cp, 3))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = BsplineComp(
        num_cp=num_cp,
        num_pt=num_pt,
        jac=jac,
        in_name='cp',
        out_name='pt',
    )
    prob.model.add_subsystem('bspline_comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    # print(prob['cp'])
    # print(prob['pt'])
    prob.check_partials(compact_print=True)
    # print(prob['cp'].shape)
    # print(prob['pt'].shape)