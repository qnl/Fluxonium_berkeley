import gdspy

import numpy as np
import ruamel.yaml as yaml

import qnldraw as qd
import qnldraw.junction as qj


from qnldraw import shapes, components

class JunctionArray(components.Component):
    __draw__ = True
    
    def draw(self, n, overlap, wire, undercut):
        jx, jy = overlap
        wx, wy = wire
        
        spacing = jy + wy
        
        positions = (np.arange(n) - (n-1)/2)*spacing
        
        for y in positions:
            junction_rect = shapes.Rectangle(jx, jy).translate(0, y)
            undercuts = [
                shapes.Rectangle(undercut_after_JJ, jy, 'right', layer=1).translate(-jx/2, y),
                shapes.Rectangle(undercut_after_JJ, jy, 'left', layer=1).translate(jx/2, y)
            ]
            
            self.add(junction_rect)
            self.add(undercuts)
            
        for i, y in enumerate(positions[:-1]):
            wire_rect = shapes.Rectangle(wx, wy, 'bottom').translate(0, y + 0.5*jy)
            undercuts = shapes.Rectangle(
                undercut, wy, 'bottom left' if i%2 else 'bottom right', layer=1
            ).translate(
                (2*(i%2) - 1)*wx/2, y + 0.5*jy
            )

            self.add(wire_rect)
            self.add(undercuts)
            
        nodes = {
            'wire1': np.array((0, positions[-1] + 0.5*jy)),
            'wire2': np.array((0, positions[0] - 0.5*jy))
        }
        return nodes

if __name__ == "__main__":
    lib = gdspy.GdsLibrary()
    chip = qd.Chip()

    with open('parameters.yaml', 'r') as f:
        params = qd.Params(yaml.load(f, yaml.CSafeLoader))

    ## These should probably be put in parameters.yaml
    n =  params['evaporation.junctions.number']
    overlap = params['evaporation.junctions.overlap'] # lx, ly, junction size
    wire = params['evaporation.junctions.wire']   # lx, ly, connecting piece between junctions
    undercut = params['evaporation.junctions.undercut']     # undercut for wires
    undercut_after_JJ = params['evaporation.junctions.undercut_after_JJ'] # undercut after junction

    dose_layers = [1, 2]
    evap_layers = params['evaporation.layers']

    junction_array = JunctionArray(n, overlap, wire, undercut)

    chip.add_component(junction_array, cid='JJArray', layers=dose_layers)

    ## Simulate Evaporation
    polys = junction_array.get_polygons(by_spec=True)
    highdose = gdspy.PolygonSet(polys[(dose_layers[0], 0)])
    lowdose = gdspy.PolygonSet(polys[(dose_layers[1], 0)])

    evaporated = qj.simulate_evaporation(
        lowdose, highdose, **params['evaporation']
    )

    for i, (layer, evap) in enumerate(zip(evap_layers, evaporated)):
        chip.add_component(evap, f'evap_{i}', layers=layer)

    cells = chip.render(name='jj_array', draw_border=False)

    lib.write_gds('jj_array_test.gds', cells=cells)