Simultaneous optimization
=========================

Installing
----------
To install CTR optimization tool and run topology optimization problems, users need to follow these steps:

1.  Install `OpenMDAO <https://openmdao.org/>`_: 
  - ``pip install 'openmdao[all]'``

2. Install ``Ozone``:

  - Use the command ``git clone`` to `this repository <https://github.com/hwangjt/ozone.git>`_,
  - Use the command ``pip install -e .`` to install ``Ozone`` .

3. Install ``Open3D``:
    Supported operating system:
    - Ubuntu 18.04+
    - macOS 10.14+
    - Windows 10 (64-bit)
    ``pip install open3d``

Other recommandations: while the ``scipy`` optimizer in OpenMDAO works for some small-scale problems, we recommend `IPOPT <https://github.com/coin-or/Ipopt/>`_ or `SNOPT <http://ccom.ucsd.edu/~optimizers/downloads/>`_.
Note that ``SNOPT`` is a commercial optimizer.



.. toctree::
  :maxdepth: 2
  :titlesonly:
