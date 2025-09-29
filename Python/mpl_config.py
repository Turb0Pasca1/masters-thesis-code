import matplotlib.pyplot as plt
import matplotlib as mpl

fsize       = 12        # fontsize
leg_fsize   = 8         # fontsize of legend
tdir        = 'in'      # tick direction
major       = 5         # major tick size
minor       = 3         # minor tick size
lwidth      = .8        # linewidth of plot surroundings
gwidth      = 2         # linewidth of graphs; 2 for reports, 3 for slides
lhandle     = 1.0       # length of line in legend
subwidth    = 4         # subplot width
subheight   = 3         # subplot height

mpl.rcParams.update({
    # font
    'font.size'                     : fsize,
    'font.family'                   : 'DejaVu Sans Mono',
    'axes.unicode_minus'            : False,
    # ticks of axes
    'xtick.minor.visible'           : True,
    'xtick.direction'               : tdir,
    'xtick.minor.bottom'            : True,
    'xtick.major.size'              : major,
    'xtick.minor.size'              : minor,
    'ytick.minor.visible'           : True,
    'ytick.direction'               : tdir,
    'ytick.minor.left'              : True,
    'ytick.major.size'              : major,
    'ytick.minor.size'              : minor,
    # general style
    'xaxis.labellocation'           : 'center',
    'yaxis.labellocation'           : 'center',
    'axes.linewidth'                : lwidth,
    'axes.grid'                     : True,
    'axes.grid.which'               : 'both',   
    'lines.linewidth'               : gwidth,       
    'figure.constrained_layout.use' : True,
    # legend
    'legend.fontsize'               : leg_fsize,
    'legend.handlelength'           : lhandle,
    'legend.loc'                    : 'best'
})

def apply_custom_grid(fig=None, show_major=True, show_minor=True, **style_kwargs):

    if fig is None:
        fig = plt.gcf()

    major_style = {
        'color'             : 'black',
        'linestyle'         : '-',
        'linewidth'         : .7,
        'alpha'             : .5,
    }
    minor_style = {
        'color'             : 'gray',
        'linestyle'         : ':',
        'linewidth'         : .5,
        'alpha'             : .5,
    }

    # by importing apply_custom_grid one can change the standard minor and major grid styles by using the kwargs in the function call minor_ ...
    for key, value in style_kwargs.items():
        if key.startswith('major_'):
            major_style[key[6:]] = value
        elif key.startswith('minor_'):
            minor_style[key[6:]] = value

    for ax in fig.get_axes():
        try:
            ax.minorticks_on()
        except Exception:
            pass  # exception for e.g. 3D axes

        if show_major:
            ax.grid(True, which='major', **major_style)
        if show_minor:
            ax.grid(True, which='minor', **minor_style)

_original_figure = plt.figure
_original_subplots = plt.subplots

def figure_with_custom_grid(*args, **kwargs):
    use_custom_grid = kwargs.pop('use_custom_grid', True)
    fig = _original_figure(*args, **kwargs)
    if use_custom_grid:
        apply_custom_grid(fig)
    return fig

def subplots_with_custom_grid(*args, **kwargs):
    use_custom_grid = kwargs.pop('use_custom_grid', True)
    scale_subplot_size = kwargs.pop('scale_subplot_size', True)
    subplot_width = kwargs.pop('subplot_width', subwidth)
    subplot_height = kwargs.pop('subplot_height', subheight)
    nrows = kwargs.get('nrows', 1)
    ncols = kwargs.get('ncols', 1)

    if scale_subplot_size and 'figsize' not in kwargs:
        total_width = ncols * subplot_width
        total_height = nrows * subplot_height
        kwargs['figsize'] = (total_width, total_height)

    fig, axes = _original_subplots(*args, **kwargs)
    if use_custom_grid:
        apply_custom_grid(fig)
    return fig, axes

plt.figure = figure_with_custom_grid
plt.subplots = subplots_with_custom_grid