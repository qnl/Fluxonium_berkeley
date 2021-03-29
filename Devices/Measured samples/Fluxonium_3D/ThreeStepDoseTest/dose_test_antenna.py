import os
import re
import datetime as dt
import gdspy
import numpy as np
import ruamel.yaml as yaml

import qnldraw as qd
import qnldraw.library as qlib

from qnldraw import components, paths, shapes, Chip, Params


class FluxoniumSymmetricAntenna(components.Component):
    """
    Capacitor pads for QNL's standard floating qubits.

    ![FloatingPads](images/FloatingPads.svg)
    Node | Name         | Node  | Name
    ---- | -------------| ----- | ----
    0    | `origin`     | 6     | `pad2_bottom`
    1    | `pad1`       | 7     | `cutout_left`
    2    | `pad2`       | 8     | `cutout_right`
    3    | `pad1_top`   | 9     | `cutout_top`
    4    | `pad2_top`   | 10    | `cutout_bottom`
    5    | `pad1_bottom`|
    """

    __draw__ = True

    def draw(self, pads, leads, spacing, cutout):
        """Draws the floating pads.
        Note that pads can be a list of two [`Params`][qnldraw.Params]
        (dictionaries) in which case the two dictionaries will refer to the left
        (`pad1`) and right (`pad2`) pads respectively.
        Args:
            pads.x (float): The `x` dimension of the pads
            pads.y (float): The `y` dimension of the pads
            spacing (float): The distance between the two pads
            cutout.x (float): The `x` dimension of the outer cutout rectangle.
            cutout.y (float): The `y` dimension of the outer cutout rectangle.

        Returns:
            dict: The locations of the nodes specified above.
        """
        if isinstance(pads, list):
            pad1, pad2 = pads
        else:
            pad1, pad2 = pads, pads
        lead = leads

        del self.params['pads']
        self.params['pad1'] = pad1
        self.params['pad2'] = pad2
        self.params['lead'] = lead

        q1x, q1y = pad1['x'], pad1['y']
        q2x, q2y = pad2['x'], pad1['y']
        q3x, q3y = lead['x'], lead['y']
        qy = max(q1y, q2y, q3y)

        qpads = [
            shapes.Rectangle(q1x, q1y, 'right').translate(-spacing / 2-q3x, 0),
            shapes.Rectangle(q2x, q2y, 'left').translate(spacing / 2+q3x, 0),
            shapes.Rectangle(q3x, q3y, 'right').translate(-spacing / 2, 0),
            shapes.Rectangle(q3x, q3y, 'left').translate(spacing / 2, 0)
        ]

        cx, cy = cutout['x'], cutout['y']
        try:
            offset = np.array(cutout['offset'])
        except KeyError:
            offset = np.zeros(2)

        cutout = shapes.Rectangle(cx, cy).translate(*offset)

        self.add(qd.boolean(cutout, qpads, 'not'))

        self.add_cutout(cutout)

        nodes = {
            'pad1': np.array((-spacing / 2 - q1x, 0)),
            'pad2': np.array((spacing / 2 + q2x, 0)),
            'pad1_top': np.array((-spacing / 2 - q1x / 2, qy / 2)),
            'pad2_top': np.array((spacing / 2 + q2x / 2, qy / 2)),
            'pad1_bottom': np.array((-spacing / 2 - q1x / 2, -qy / 2)),
            'pad2_bottom': np.array((spacing / 2 + q2x / 2, -qy / 2)),
            'cutout_left': np.array((-cx / 2 + offset[0], 0)),
            'cutout_right': np.array((cx / 2 + offset[0], 0)),
            'cutout_top': offset + (0, cy / 2),
            'cutout_bottom': offset + (0, -cy / 2)
        }

        return nodes
cell_x = 7000
cell_y = 4000
num_qubits_x = int(42000/cell_x)
num_qubits_y = int(20000/cell_y)
chip_x = cell_x*num_qubits_x
chip_y = cell_y*num_qubits_y


def draw_qubits(chip, qparams):
    ## Construct a qubit pads component
    Q = FluxoniumSymmetricAntenna(**qubit_params['antenna'])

    xs = np.linspace(-chip_x/2+cell_x/2, chip_x/2 - cell_x/2, num_qubits_x)
    ys = np.linspace(-chip_y / 2 + cell_y/2, chip_y / 2 - cell_y/2, num_qubits_y)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Qij = Q.place(location=(x, y))
            chip.add_component(Qij, cid=f'Q{i*num_qubits_y+j:02d}', layers=qubit_params['layer'])

def draw_markers(chip, qparams):
    ## Construct a qubit pads component
    M = qlib.FloatingPads(**qubit_params['marker'])

    xs = np.linspace(-chip_x / 2, chip_x / 2 , num_qubits_x+1)
    ys = np.linspace(-chip_y / 2 , chip_y / 2 , num_qubits_y+1)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Mij = M.place(location=(x, y))
            chip.add_component(Mij, cid=f'M{i * num_qubits_y + j:02d}', layers=qubit_params['layer'])

if __name__ == '__main__':
    start_time = dt.datetime.now()
    param_dir = '/Users/longnguyen/Documents/GitHub/Fluxonium_berkeley/Devices/Measured samples/Fluxonium_3D/ThreeStepDoseTest/parameters'
    outfile = '3D_dosetest_Nb.gds'
    gds_dir = '/Users/longnguyen/Documents/GitHub/Fluxonium_berkeley/Devices/Measured samples/Fluxonium_3D/ThreeStepDoseTest/pattern_files/'

    with open(os.path.join(param_dir, 'transmon_v5.yaml'), 'r') as f:
        qubit_params = Params(yaml.load(f, yaml.CSafeLoader))

    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6  # um
    lib.precision = 1.0e-9  # nm

    chip = Chip(size = [chip_x,chip_y])
    draw_qubits(chip, qubit_params)
    draw_markers(chip, qubit_params)
    group = ['READOUTRES00', 'BUS', 'BUSLINE']

    intersections = [
        ([cid for cid in chip.components if re.match(r'^Q\d+', cid)] +
         [cid for cid in chip.components if re.match(r'^COUPLING', cid)])
    ]

    # Render mask
    mask = chip.render('MASK', intersections = intersections, draw_border = True)

    # Selective Rendering
    # mask = chip.render('MASK', intersections=intersections, group=group, draw_border=True)

    # Simulation Rendering
    # mask = chip.render(
    #     'MASK',
    #     intersections=intersections,
    #     draw_border=True,
    #     group=group,
    #     sim=(0, 11),
    #     draw_substrate=True
    # )
    lib.write_gds(os.path.join(gds_dir, outfile), cells=mask)