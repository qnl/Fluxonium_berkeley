import os
import re
import datetime as dt
import gdspy
import numpy as np
import ruamel.yaml as yaml

import qnldraw as qd
import qnldraw.library as qlib

from qnldraw import components, paths, shapes, Chip, Params

from sample_components import (
    ReadoutBus,
    CouplingResonator,
    HorizontalCouplingResonator,
    ReadoutResonator,
    ControlLine,
    ReadoutLine
)


def draw_multiplexed_readout(chip, mpr_params):
    bus = ReadoutBus(cpw=mpr_params['cpw'], **mpr_params['purcell_filter'])
    bus = bus.place((0, mpr_params['bus_y_center']), node='center')

    chip.add_component(bus, cid='BUS', layers=mpr_params['layer'])


def draw_qubits(chip, qparams):
    ## Construct a qubit pads component
    Q = qlib.FloatingPads(**qubit_params['qubits'])

    ## Qubit Location Parameters
    x = qubit_params['qubit_x_location']
    num_qubits = qubit_params['num_qubits']

    bus = chip.get_component('BUS')

    ## Transmon y locations
    _, bus_y_top = bus.node('qb_top')
    _, bus_y_bottom = bus.node('qb_bottom')
    ys = np.linspace(bus_y_bottom, bus_y_top, 8)

    ## Left Qubits
    for i, y in enumerate(ys[::2]):
        Qi = Q.place(location=(-x, y))
        chip.add_component(Qi, cid=f'Q{i:02d}', layers=qubit_params['layer'])

    ## Right Qubits
    for i, y in enumerate(np.flip(ys[1::2])):
        Qi = Q.place(location=(x, y), mirror='y')
        chip.add_component(
            Qi,
            cid=f'Q{i + num_qubits // 2:02d}',
            layers=qubit_params['layer']
        )


def draw_coupling_resonators(chip, coupling_params):
    ## Compute qubit indices
    N = len(coupling_params['vertical_couplers.lengths']) + 2
    qubits = list(range(N // 2 - 1)) + list(range(N // 2, N - 1))

    ## Compute coupling pad parameters
    q0 = chip.get_component('Q00')
    cpad = coupling_params['coupler_pad']
    cpad.update({
        'cpw_length': ((q0.node('cutout_top') - q0.node('pad1_top'))[1]
                       - cpad['base.width'] - cpad['gap']),
        'base.length': q0.get_params('pad1.x') + 2 * cpad['gap'] + 2 * cpad['fingers.width']
    })

    ## Draw vertical couplers first
    vcouplers = coupling_params['vertical_couplers']
    lengths = vcouplers.pop('lengths')

    ## Compute span from qubit positions
    q1 = chip.get_component('Q01')
    _, span = q1.node('cutout_bottom') - q0.node('cutout_top')

    ## Create the coupling pad component, which gets reused by all coupling
    claw = qlib.Claw(cpw=coupling_params['cpw'], **cpad)

    ## Draw all vertical coupling resonators
    for qid, L in zip(qubits, lengths):
        qi = chip.get_component(f'Q{qid:02d}')
        qj = chip.get_component(f'Q{qid + 1:02d}')

        if qid < N // 2:
            rotation, offset, n = -90, (0, cpad['gap']), f'pad{(qid + 1) % 2 + 1}_top'
        else:
            rotation, offset, n = 90, (0, -cpad['gap']), f'pad{(N - qid - 1) % 2 + 1}_bottom'

        cres = CouplingResonator(
            cpw=coupling_params['cpw'],
            span=span,
            Ltotal=L,
            coupler=claw,
            **vcouplers
        )

        cres = cres.attach(qi, n, 'right', offset=offset, rotation=rotation)
        chip.add_component(cres, cid=f'COUPLINGRESONATOR{qid}{qid + 1}', layers=coupling_params['layer'])

    ## Draw horizontal coupling resonators now
    hcouplers = coupling_params['horizontal_couplers']
    lower = hcouplers['lower']
    upper = hcouplers['upper']

    qleft = chip.get_component(f'Q00')
    qright = chip.get_component(f'Q{N - 1:02d}')

    lower['meander.ys'] -= np.array(lower['y_position'])

    lcoupler_cmp = HorizontalCouplingResonator(
        cpw=coupling_params['cpw'],
        lx=(qright.node('pad1_bottom') - qleft.node('pad1_bottom'))[0],
        ly1=qleft.node('cutout_bottom')[1] - lower['y_position'],
        ly2=qright.node('cutout_bottom')[1] - lower['y_position'],
        radius=hcouplers['radius'],
        meander=lower['meander'],
        coupler=claw,
    )

    lower_coupler = lcoupler_cmp.attach(qleft, 'pad1_bottom', 'left', offset=(0, -cpad['gap']))
    chip.add_component(lower_coupler, cid=f'COUPLINGRESONATOR{N}0', layers=coupling_params['layer'])

    qleft = chip.get_component(f'Q{N // 2 - 1:02d}')
    qright = chip.get_component(f'Q{N // 2:02d}')

    upper['meander.ys'] = np.array(upper['y_position']) - upper['meander.ys']
    ucoupler_cmp = HorizontalCouplingResonator(
        cpw=coupling_params['cpw'],
        lx=(qright.node('pad1_top') - qleft.node('pad1_top'))[0],
        ly1=upper['y_position'] - qleft.node('cutout_top')[1],
        ly2=upper['y_position'] - qright.node('cutout_top')[1],
        radius=hcouplers['radius'],
        meander=upper['meander'],
        coupler=claw
    )
    upper_coupler = ucoupler_cmp.attach(qleft, 'pad1_top', 'left', offset=(0, cpad['gap']), mirror='x')
    chip.add_component(upper_coupler, cid=f'COUPLINGRESONATOR{N // 2 - 1}{N // 2}', layers=coupling_params['layer'])


def draw_readout_resonators(chip, readout_params):
    ## Compute readout resonator lengths
    N = len(readout_params['sep_bus_couplers'])
    rfreq, rlen = readout_params['readout_res_length'], readout_params['readout_res_freq']
    rmin_freq, rmax_freq = readout_params['readout_res_freq_min'], readout_params['readout_res_freq_max']
    rlengths = 1 / np.linspace(rmin_freq / (rfreq * rlen), rmax_freq / (rfreq * rlen), N)

    ## Important note. The readout resonator lengths were computed incorrectly
    #  in the MaskV5 code, leading to the lengths being off by 2*radius. In the original
    #  v6 mask code, the choice was made to keep the same incorrect lengths. We've
    #  similarly compensated for this error here by adding 2*radius to the total length
    #  passed to the ReadoutResonator component.
    #
    #  See https://github.com/qnl/Mask_Design/blob/master/MultiQubitMask_v6/mask4_library.py#L591
    rlengths += 2 * readout_params['radius']

    ## Coupling Pad parameters
    cpw = readout_params['cpw']
    qb0 = chip.get_component('Q00')

    cpad = qlib.CouplingPad(
        cpw=cpw,
        pad={
            'length': 0.9 * qb0.get_params('cutout.y') - 2 * cpw['gap'],
            'width': 2 * cpw['width']
        },
        gap=cpw['gap'],
        cpw_length=qb0.get_params('cutout.x') + 100,
        name='RCOUPLER'
    )

    ## Readout resonator parameters
    bus = chip.get_component('BUS')

    qb_bus_distance = ((bus.node('origin') - qb0.node('cutout_right'))[0]
                       - bus.get_params('cpw.width') / 2
                       - bus.get_params('cpw.gap'))

    for i, (qb_sep, bus_sep, bus_rev, L) in enumerate(zip(
            readout_params['qubit_coupler_separation'],
            readout_params['sep_bus_couplers'],
            readout_params['reverse_bus_couplers'],
            rlengths
    )):
        if i < N // 2:
            mirror = 'x' if bus_rev else None
            rotation = 0
            offset = (qb_sep + cpad.get_params('gap'), 0)
        else:
            # 180 deg rotation is equivalent to reflection about x, then y
            mirror, rotation = (None, 180) if bus_rev else ('y', 0)
            offset = (-qb_sep - cpad.get_params('gap'), 0)

        resonator = ReadoutResonator(
            cpw=cpw,
            L=L,
            lx=qb_bus_distance - qb_sep - bus_sep,
            l_bus_coupler=readout_params['l_bus_coupler'],
            meander_gap=readout_params['meander_gap'],
            radius=readout_params['radius'],
            cpad=cpad
        ).attach(
            chip.get_component(f'Q{i:02d}'),
            'cutout_right',
            'RCOUPLER.pad',
            offset=offset,
            rotation=rotation,
            mirror=mirror
        )

        chip.add_component(resonator, cid=f'READOUTRES{i:02d}', layers=readout_params['layer'])

def draw_purcell_resonators(chip, readout_params):
    ## Compute readout resonator lengths
    N = len(readout_params['sep_bus_couplers'])
    rfreq, rlen = readout_params['readout_res_length'], readout_params['readout_res_freq']
    rmin_freq, rmax_freq = readout_params['readout_res_freq_min'], readout_params['readout_res_freq_max']
    rlengths = 1 / np.linspace(rmin_freq / (rfreq * rlen), rmax_freq / (rfreq * rlen), N)

    ## Important note. The readout resonator lengths were computed incorrectly
    #  in the MaskV5 code, leading to the lengths being off by 2*radius. In the original
    #  v6 mask code, the choice was made to keep the same incorrect lengths. We've
    #  similarly compensated for this error here by adding 2*radius to the total length
    #  passed to the ReadoutResonator component.
    #
    #  See https://github.com/qnl/Mask_Design/blob/master/MultiQubitMask_v6/mask4_library.py#L591
    rlengths += 2 * readout_params['radius']

    ## Coupling Pad parameters
    cpw = readout_params['cpw']
    qb0 = chip.get_component('Q00')

    cpad = qlib.CouplingPad(
        cpw=cpw,
        pad={
            'length': 0. * qb0.get_params('cutout.y') - 2 * cpw['gap'],
            'width': 0 * cpw['width']
        },
        gap=cpw['gap'],
        cpw_length=qb0.get_params('cutout.x') + 100,
        name='RCOUPLER'
    )

    ## Readout resonator parameters
    bus = chip.get_component('BUS')

    qb_bus_distance = ((bus.node('origin') - qb0.node('cutout_right'))[0]
                       - bus.get_params('cpw.width') / 2
                       - bus.get_params('cpw.gap'))

    for i, (qb_sep, bus_sep, bus_rev, L) in enumerate(zip(
            readout_params['qubit_coupler_separation'],
            readout_params['sep_bus_couplers'],
            readout_params['reverse_bus_couplers'],
            rlengths
    )):
        if i < N // 2:
            mirror = 'x' if bus_rev else None
            rotation = 0
            offset = (qb_sep + cpad.get_params('gap'), 0)
            # offset = (qb_sep,0)
        else:
            # 180 deg rotation is equivalent to reflection about x, then y
            mirror, rotation = (None, 180) if bus_rev else ('y', 0)
            offset = (-qb_sep - cpad.get_params('gap'), 0)
            # offset = (-qb_sep, 0)
        resonator = ReadoutResonator(
            cpw=cpw,
            L=L,
            lx=qb_bus_distance - qb_sep - bus_sep,
            l_bus_coupler=readout_params['l_bus_coupler'],
            meander_gap=readout_params['meander_gap'],
            radius=readout_params['radius'],
            cpad=cpad
        ).attach(
            chip.get_component(f'Q{i:02d}'),
            'cutout_right',
            'RCOUPLER.pad',
            offset=offset,
            rotation=rotation,
            mirror=mirror
        )

        chip.add_component(resonator, cid=f'PURCELLRES{i:02d}', layers=readout_params['layer'])

def draw_control_lines(chip, control_params):
    cpw = control_params['cpw']
    N = control_params['num_ctrl_lines_per_side'] * 2

    ## Creating the launch component
    launch = qlib.Launch(
        cpw=cpw,
        taper_length=control_params['bondpad'].pop('taper'),
        bondpad=control_params['bondpad']
    )

    ## Compute control line ys
    control_line_ys = np.linspace(
        control_params['ctrl_line_y_start'],
        control_params['ctrl_line_y_end'],
        control_params['num_ctrl_lines_per_side'] + 1
    )

    ## Compute intersection position
    intersection = (control_params['intersection_start_x']
                    + control_params['intersection_end_x']) / 2

    ## Reorder by Qubit ID (0 indexed, clockwise, starting from lower left)
    control_line_ys = np.concatenate([control_line_ys[:-1], control_line_ys[1:][::-1]])

    x_left = chip.extent()[0][0] + control_params['ctrl_line_x_offset']
    x_right = chip.extent()[0][1] - control_params['ctrl_line_x_offset']

    ## Make ControlLines and add to chip
    for i, y in enumerate(control_line_ys):
        rotation, sign, x = (0, -1, x_left) if i < N // 2 else (180, 1, x_right)

        qb = chip.get_component(f'Q{i:02d}')

        start = np.array((x, y))
        end = qb.node('cutout_left') + (sign * control_params['qubit_separation'], 0)

        ctrl_line = ControlLine(cpw, launch, start, end, intersection, rotation).place()

        chip.add_component(ctrl_line, cid=f'CONTROLLINE{i:02d}', layers=control_params['layer'])


def draw_readout_launch(chip, mpr_params, qubit_params):
    N = qubit_params['num_qubits']
    Qn = chip.get_component(f'Q{N - 1:02d}')
    mpr_params['hero_bond.x'] = Qn.node('pad1_bottom')[0]

    launch = qlib.Launch(cpw=mpr_params['cpw'], **mpr_params['readout_launch'])

    keys = ['cpw', 'hero_bond', 'shift_x_position', 'radius']
    rline_params = Params(
        {k: v for k, v in mpr_params.items() if k in keys}
    )

    bus = chip.get_component('BUS')

    # Compute the launch x position
    launch_x = chip.extent()[0, 1] - mpr_params['launch_x_offset']

    readout_line = ReadoutLine(
        start=bus.node('IDC.cpw1'),
        launch=launch,
        launch_position=np.array((launch_x, mpr_params['launch_y'])),
        **rline_params
    ).place()

    chip.add_component(readout_line, cid='BUSLINE', layers=mpr_params['layer'])


def draw_junctions(chip, sample_params, qubit_params):
    jparams = sample_params['junctions']

    for i, file in enumerate(jparams['files']):
        gds = {
            'file': os.path.join('./pattern_files/', file),
            'toplevel': 'junction_mask',
            'layers': jparams['original_layers']
        }

        junction = qlib.Junction(**jparams['sim'], gds=gds)

        position = chip.get_component(f'Q{i:02d}').node('origin')
        chip.add_component(
            junction.place(position),
            cid=f'J{i:02d}',
            layers=jparams['layers']
        )


if __name__ == '__main__':
    start_time = dt.datetime.now()
    param_dir = './parameters'
    outfile = 'mask6.gds'
    gds_dir = './pattern_files/'

    with open(os.path.join(param_dir, 'mqcv6.yaml'), 'r') as f:
        sample_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'transmon_v5.yaml'), 'r') as f:
        qubit_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'multiplexed_readout.yaml'), 'r') as f:
        mpr_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'coupling_resonators.yaml'), 'r') as f:
        coupler_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'readout_resonators.yaml'), 'r') as f:
        readout_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'purcell_resonators.yaml'), 'r') as f:
        purcell_params = Params(yaml.load(f, yaml.CSafeLoader))

    with open(os.path.join(param_dir, 'control_lines.yaml'), 'r') as f:
        control_params = Params(yaml.load(f, yaml.CSafeLoader))

    lib = gdspy.GdsLibrary()
    lib.unit = 1.0e-6  # um
    lib.precision = 1.0e-9  # nm

    chip = Chip()
    draw_multiplexed_readout(chip, mpr_params)
    draw_qubits(chip, qubit_params)
    # draw_coupling_resonators(chip, coupler_params)
    draw_readout_resonators(chip, readout_params)
    draw_purcell_resonators(chip, purcell_params)
    draw_control_lines(chip, control_params)
    draw_readout_launch(chip, mpr_params, qubit_params)
    # draw_junctions(chip, sample_params, qubit_params)

    group = ['READOUTRES00', 'BUS', 'BUSLINE']

    intersections = [
        ([cid for cid in chip.components if re.match(r'^Q\d+', cid)] +
         [cid for cid in chip.components if re.match(r'^COUPLING', cid)])
    ]

    # Render mask
    mask = chip.render('MASK', intersections=intersections, draw_border=True)

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

    print(f'Finished writing mask in {dt.datetime.now() - start_time}')