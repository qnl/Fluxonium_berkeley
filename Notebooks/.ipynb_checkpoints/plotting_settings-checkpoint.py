"""
A module to customize Matplotlib parameters
"""

from matplotlib import pyplot as plt
# from cycler import cycler

plt.figure(figsize=[5,4])

# plt.style.use(['seaborn-deep'])
# custom_cycler = (cycler(color=['#003262', '#C4820F','#55A868',
#                                '#d62728', '#9467bd','#CCB974',
#                                '#8c564b', '#e377c2', '#7f7f7f',
#                                '#bcbd22', '#17becf','#1f77b4']))
# plt.rc('axes', prop_cycle=custom_cycler)

plt.rc('lines',
       linewidth=2,
       dashed_pattern=[4, 3],
       dashdot_pattern=[4, 2, 1, 2],
       dotted_pattern=[1, 2])

plt.rc('font',
       family='serif',
       size=16)

plt.rc('mathtext',
       fontset='cm')

plt.rc('axes',
       linewidth=1.5,
       titlesize=16,
       labelsize=20,
       titlepad=10,
       )

plt.rc('xtick',
       top=True,
       labelsize=16,
       direction='in')

plt.rc('xtick.major',
       size=7,
       width=1.5)

plt.rc('xtick.minor',
       size=4,
       width=1.5,
       visible=True)

plt.rc('ytick',
       right=True,
       labelsize=16,
       direction='in')

plt.rc('ytick.major',
       size=7,
       width=1.5)

plt.rc('ytick.minor',
       size=4,
       width=1.5,
       visible=True)

plt.rc('legend',
       fontsize=15)

plt.rc('figure',
       titlesize=15)

plt.tick_params(labelsize = 16)
