# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import tempfile

import matplotlib
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image
import torch

import stable_solvers as solvers

N = 128


class PseudoNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pts = torch.zeros((2,))
        self.register_parameter('pts', torch.nn.Parameter(pts))

    def forward(self, x = None):
        x, y = self.pts
        return (((6 + 2 * y) * x ** 2) + y).view(1)


def make_surface(path: str):
    net = PseudoNet()

    xs, ys = torch.meshgrid(
        (torch.linspace(-1, 1, N), torch.linspace(0, 1, N)),
        indexing='ij'
    )
    xys = torch.stack((xs, ys), dim=2)
    zs = [
        torch.func.functional_call(net, {'pts': xy}, ()).view(1)
        for xy in xys.view(-1, 2)
    ]
    zs = torch.cat(zs).view(N, N)

    fig, ax = plot.subplots(subplot_kw={'projection': '3d'}, figsize=(6, 6))
    ax.plot_surface(xs, ys, zs, vmin=zs.min() * 2, cmap=matplotlib.cm.viridis)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 9)
    plot.axis('off')
    plot.savefig(path)
    plot.close()


def make_curve(path: str):
    net = PseudoNet()

    xy = torch.tensor([-0.9, 0.9])
    xys, zs = [xy.clone()], []

    loss_func = solvers.LossFunction(
        torch.utils.data.TensorDataset(torch.zeros(1), torch.zeros(1)),
        lambda x, y: x.view(1,),
        net,
    )
    solver = solvers.ExponentialEulerSolver(
        xy, loss_func, 0.001, 0,
    )

    while xys[-1][1] > 0:
        zs.append(solver.step().loss)
        xys.append(solver.params.clone())

    xys = torch.stack(xys[:-1], dim=0)
    zs = torch.tensor(zs)

    fig, ax = plot.subplots(subplot_kw={'projection': '3d'}, figsize=(6, 6))
    ax.plot(xys[:, 0], xys[:, 1], zs, color='yellow')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 9)
    plot.axis('off')
    plot.savefig(path)
    plot.close()


def make_logo(path: str):
    with tempfile.TemporaryDirectory() as temp_root:
        surf_path = os.path.join(temp_root, 'surface.png')
        make_surface(surf_path)
        curve_path = os.path.join(temp_root, 'curve.png')
        make_curve(curve_path)

        surface = np.array(Image.open(surf_path))
        curve = np.array(Image.open(curve_path))

    mask = (curve[..., 1] < 255)
    surface[mask, :] = curve[mask, :]
    Image.fromarray(surface[..., :3]).save(path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Stable Solvers'
copyright = '2024, Mark Lowell'
author = 'Mark Lowell'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

logo_path = os.path.join(__file__, '../_static/logo.png')
make_logo(logo_path)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

try:
    import branding

    html_theme_options = {}
    html_theme_options.update(branding.sphinx.SPHINX_THEME_OPTIONS)

    pygments_style = 'viridian'

except ImportError:
    print('viridian not found. Building with alabaster defaults.')


