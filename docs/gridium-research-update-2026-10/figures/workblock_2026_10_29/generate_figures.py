#!/usr/bin/env python3
"""Render authenticated saved scalars and a planned-work diagram; no physics.

Default reproduction needs only the adjacent figure_data.json and requirements.
--verify-local-sources additionally authenticates retained local source files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA_SHA256 = '2e433e95d5b218f1a5a7a1103481db35c28a59c3b9bfcaba2dd8054acee47d76'
DPI = 320
NAVY, BLUE, LIGHT_BLUE = '#18354b', '#315e82', '#7aa2bb'
RED, GRID, GREY = '#a83d36', '#e5eaf0', '#5b6873'
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'], 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5,
    'text.color': NAVY, 'axes.labelcolor': NAVY,
    'xtick.color': GREY, 'ytick.color': GREY,
    'svg.fonttype': 'none', 'svg.hashsalt': 'gridium-oct29-2026',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(verify_sources):
    path = HERE / 'figure_data.json'
    assert sha(path) == DATA_SHA256, 'Bundled scalar provenance changed'
    data = json.loads(path.read_text())
    f1 = data['figure1']
    assert len(f1['rows']) == 4 and len(f1['carrier_state_pairs']) == 9
    for row in f1['rows']:
        changes = row['saved_lower_minus_upper_carrier_changes_MHz']
        assert row['maximum_absolute_carrier_change_MHz'] == max(map(abs, changes.values()))
        assert row['tracking_safe'] is True and row['upper_k'] <= 380
    f2 = data['figure2']
    assert f2['minimum_available_proxy_bytes'] == 12 * 2**30
    assert f2['minimum_free_physical_bytes'] == 2**30
    for row in f2['sessions']:
        assert row['free_pages'] * row['page_bytes'] == row['free_physical_bytes']
        assert sum(row[k] for k in ('free_pages', 'inactive_pages', 'speculative_pages')) * row['page_bytes'] == row['available_proxy_bytes']
    if verify_sources:
        root = HERE.parents[3]
        for name, expected in data['source_hashes'].items():
            assert sha(root / name) == expected, 'Source changed: ' + name
        metrics = json.loads((root / 'research/sandbox/epsj011_k380_preregistration/k380_metrics.json').read_text())
        for row in f1['rows']:
            components = metrics['comparisons'][row['comparison']]['components']
            assert row['saved_lower_minus_upper_carrier_changes_MHz'] == {
                k.split(':')[1]: v['value'] for k, v in components.items() if k.startswith('carrier:')}
    return data


def base_title(fig, title, subtitle):
    fig.text(.075, .955, title, fontsize=18, weight='bold', va='top')
    fig.text(.075, .885, subtitle, fontsize=12.5, color=GREY, va='top')


def clean_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, linewidth=.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#bac5ce')
    ax.tick_params(length=0, pad=8)


def figure1(data):
    rows = data['figure1']['rows']
    fig = plt.figure(figsize=(9, 5.5))
    base_title(fig, 'The retained-basis carrier tail is unresolved',
               'PRIOR SCIENTIFIC EVIDENCE · Same k380 parent · Fixed N71 · εJ = 0.11')
    ax = fig.add_axes([.13, .24, .83, .55])
    values = [r['maximum_absolute_carrier_change_MHz'] for r in rows]
    ax.bar(range(4), values, width=.57, color=BLUE, zorder=3)
    for x, value in enumerate(values):
        ax.text(x, value + .22, f'{value:.6f}', ha='center', fontsize=13, weight='bold')
    ax.set_xticks(range(4), [f"k{r['lower_k']} to k{r['upper_k']}" for r in rows])
    ax.set_xlim(-.6, 3.6)
    ax.set_ylim(0, 8.2)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_ylabel('Maximum absolute carrier change (MHz)', labelpad=12)
    ax.set_xlabel('Nested Stage-2 retained-prefix increment', labelpad=12)
    clean_axes(ax)
    fig.text(.075, .055, 'Frozen retained-basis verdict: FAIL · No k400/k420 observations',
             fontsize=12, color=RED)
    return fig, []


def figure2(data):
    f2 = data['figure2']
    fig = plt.figure(figsize=(10, 5.5))
    base_title(fig, 'Both observed sessions failed host memory admission',
               'Apple M3 Pro · 18 GiB unified memory · 29 October 2026')
    specs = [(.10, .36, 'available_proxy_bytes', 12, 14.4,
              '(a) Canonical availability proxy', [0, 4, 8, 12]),
             (.62, .34, 'free_physical_bytes', 1, 1.2,
              '(b) Physically free memory', [0, .4, .8, 1.2])]
    for left, width, field, gate, ymax, title, ticks in specs:
        ax = fig.add_axes([left, .24, width, .52])
        values = [r[field] / 2**30 for r in f2['sessions']]
        ax.bar([0, 1], values, width=.52, color=[BLUE, LIGHT_BLUE], zorder=3)
        for x, value in enumerate(values):
            ax.text(x, value + .028 * ymax, f'{value:.6f}', ha='center',
                    fontsize=12.5, weight='bold')
        ax.axhline(gate, color=RED, linewidth=1.8, linestyle=(0, (5, 3)), zorder=4)
        ax.text(.5, gate + .045 * ymax, f'Required ≥{gate} GiB', color=RED,
                fontsize=12, weight='bold', ha='center')
        ax.set_title(title, loc='left', pad=14, weight='bold')
        ax.set_ylabel('GiB', labelpad=8)
        ax.set_xticks([0, 1], ['First\n10:02 PDT', 'Quiet-session\n10:08 PDT'])
        ax.set_xlim(-.65, 1.65)
        ax.set_ylim(0, ymax)
        ax.set_yticks(ticks)
        clean_axes(ax)
    fig.text(.075, .055, 'Both gates must pass. Pressure percentages are not substitutes.', fontsize=12)
    return fig, []


def figure3(data):
    assert data['figure3']['environment_admission'] == 'CURRENTLY BLOCKED'
    fig = plt.figure(figsize=(10, 8.6))
    base_title(fig, 'The k420 experiment remains planned, not executed',
               'Real execution requires separate authorization · Metadata tests are not science')
    ax = fig.add_axes([0, 0, 1, .86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    boxes = []

    def box(x, y, w, h, title, body, *, blocked=False, planned=False, title_size=13):
        color = RED if blocked else BLUE
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.007,rounding_size=.009',
                               facecolor='#fff4f1' if blocked else '#f2f6f9',
                               edgecolor=color, linewidth=1.25,
                               linestyle=(0, (4, 3)) if planned else '-')
        ax.add_patch(patch)
        t1 = ax.text(x + w/2, y + h*.76, title, ha='center', va='center',
                     fontsize=title_size, weight='bold', color=color)
        t2 = ax.text(x + w/2, y + h*.31, body, ha='center', va='center',
                     fontsize=11.5, color=NAVY, linespacing=1.3)
        boxes.extend([(t1, patch), (t2, patch)])

    def arrow(start, end, *, blocked=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                     linewidth=1.35, color=RED if blocked else BLUE,
                     linestyle=(0, (4, 3)) if blocked else '-', shrinkA=2, shrinkB=2))

    box(.27, .87, .46, .092, 'Authenticated k380 evidence', 'PREVIOUSLY COMPUTED')
    arrow((.5, .862), (.5, .815))
    box(.27, .687, .46, .12, 'Execution environment + admission',
        'External compute access: UNVERIFIED')
    box(.018, .675, .22, .146, 'Local memory',
        'Both observed\nsessions failed', blocked=True)
    box(.762, .658, .22, .18, 'Protected boundary',
        'Unavailable under the\nrequired same-user\nwriter threat model', blocked=True, title_size=12)
    arrow((.245, .747), (.263, .747), blocked=True)
    arrow((.755, .747), (.737, .747), blocked=True)
    arrow((.5, .679), (.5, .578), blocked=True)
    ax.text(.5, .631, 'CURRENTLY BLOCKED', ha='center', va='center',
            color=RED, fontsize=12, weight='bold',
            bbox={'facecolor':'white', 'edgecolor':'none', 'pad':4})
    box(.27, .475, .46, .094, 'Stage A — one genuine k420 parent',
        'PLANNED · no scientific execution', planned=True)
    arrow((.5, .467), (.5, .422))
    box(.27, .318, .46, .094, 'Stage B — ten frozen Stage-2 cases',
        'PLANNED · fixed protocol and quality gates', planned=True)
    arrow((.5, .310), (.5, .265))
    box(.27, .161, .46, .094, 'Stage C — convergence assessment',
        'PLANNED · k380 to k400 to k420', planned=True)
    arrow((.5, .153), (.5, .112))
    box(.12, .012, .76, .090, 'Output: a tested convergence verdict',
        'PASS / BORDERLINE / FAIL — not an assumed converged spectrum')
    return fig, boxes


def save(fig, boxes, stem, qa_dir):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        b = text.get_window_extent(renderer)
        assert b.x0 >= 1 and b.y0 >= 1 and b.x1 <= canvas.width-1 and b.y1 <= canvas.height-1, 'Text outside canvas: ' + text.get_text()
    for text, patch in boxes:
        b, boundary = text.get_window_extent(renderer), patch.get_window_extent(renderer)
        assert b.x0 >= boundary.x0+2 and b.x1 <= boundary.x1-2 and b.y0 >= boundary.y0+2 and b.y1 <= boundary.y1-2, 'Text outside box: ' + text.get_text()
    png, svg = HERE / (stem + '.png'), HERE / (stem + '.svg')
    fig.savefig(png, dpi=DPI, metadata={'Title':stem, 'Description':'Saved scalar data SHA-256: '+DATA_SHA256})
    fig.savefig(svg, metadata={'Date':'2026-10-29', 'Creator':'Gridium documentation figures',
                               'Description':'Saved scalar data SHA-256: '+DATA_SHA256})
    tree = ET.parse(svg)
    assert len(tree.findall('.//{http://www.w3.org/2000/svg}text')) >= 15
    assert not tree.findall('.//{http://www.w3.org/2000/svg}image'), 'SVG must remain vector'
    with Image.open(png) as im:
        width, height = im.size
        dpi = im.info.get('dpi')
        assert dpi and min(dpi) >= 300 and width >= 2500
        if qa_dir:
            qa_dir.mkdir(parents=True, exist_ok=True)
            im.resize((900, round(height*900/width)), Image.Resampling.LANCZOS).save(qa_dir/(stem+'_900px.png'))
    if qa_dir:
        import cairosvg
        cairosvg.svg2png(url=str(svg), write_to=str(qa_dir/(stem+'_svg_900px.png')),
                        output_width=900)
    plt.close(fig)
    return {'stem':stem, 'pixel_size':[width,height], 'dpi':dpi,
            'png_sha256':sha(png), 'svg_sha256':sha(svg),
            'text_within_canvas':True, 'diagram_text_within_boxes':True,
            'svg_vector_text':True, 'shared_data_sha256':DATA_SHA256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-local-sources', action='store_true')
    parser.add_argument('--qa-dir', type=Path)
    args = parser.parse_args()
    data = load_data(args.verify_local_sources)
    outputs = []
    for make, stem in [(figure1, 'fig01_retained_basis_convergence'),
                       (figure2, 'fig02_host_memory_admission'),
                       (figure3, 'fig03_k420_execution_roadmap')]:
        fig, boxes = make(data)
        outputs.append(save(fig, boxes, stem, args.qa_dir))
    record = {'source_data_sha256':DATA_SHA256, 'local_sources_verified':args.verify_local_sources,
              'matplotlib_version':matplotlib.__version__, 'png_dpi_requested':DPI,
              'outputs':outputs, 'new_scientific_calculations':0,
              'visual_inspection':'Recorded separately in FIGURE_VALIDATION.md; automated bounds are not visual inspection.'}
    (HERE/'RENDER_CHECKS.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'figures_created':len(outputs), 'numerical_and_geometry_checks':'PASS'}))


if __name__ == '__main__':
    main()
