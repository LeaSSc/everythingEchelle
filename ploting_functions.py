import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as c
#%% plot format 14.06.25

#both colums
# sSize = 10
# mSize = 12
# lSize = 16

# one column
# sSize = 14
# mSize = 16
# lSize = 20

# sSize = 12
# mSize = 14
# lSize = 18

# sSize = 8
# mSize = 10
# lSize = 14

def plot_format(color=False
               ,sSize=10,mSize=12,lSize=16
               ,axcolor=False,bold=False
               ,transparent_ax_facecolor=False):
    #plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('default')
    
    plt.rcParams['savefig.dpi'] = 300
    
    plt.rc('font', size=sSize) 
    plt.rc('axes', titlesize=lSize,labelsize=mSize)     
    plt.rc('xtick', labelsize=sSize)  
    plt.rc('ytick', labelsize=sSize)   
    plt.rc('legend', fontsize=sSize)   
    plt.rc('figure', titlesize=lSize, figsize= (10,6))
    
    plt.rcParams['axes.xmargin'] = 0.02
    plt.rcParams['axes.ymargin'] = 0.05
    
    #plt.rcParams["axes.axisbelow"] = False
    
    # plt.rcParams['path.simplify'] = True
    # plt.rcParams['path.simplify_threshold'] = 0.5
    
    if color:
        #plt.rcParams['axes.edgecolor'] = color
        plt.rcParams['xtick.color'] = color
        plt.rcParams['ytick.color'] = color
        plt.rcParams['text.color'] = 'k'
        plt.rcParams['axes.labelcolor'] = color
        
        plt.rcParams['axes.facecolor'] = 'w'#
        plt.rcParams['figure.facecolor'] = (0.0, 0.0, 0.0, 0.0)
    # else:
    #     plt.rcParams['figure.facecolor'] = (1.0, 1.0, 1.0, 0.0)
    
    if axcolor:
        plt.rcParams['axes.edgecolor'] = axcolor  
    
    if transparent_ax_facecolor:
        plt.rcParams['axes.facecolor'] = (0.0, 0.0, 0.0, 0.0)
        
    if bold:
        plt.rcParams.update({"font.weight": "bold"})
        plt.rcParams.update({"axes.labelweight": "bold"})
        plt.rcParams.update({"axes.titleweight": "bold"})
            
def RGB_to_hex(color):
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    return hex_color
                
def rasterize_all_lines(fig=False):
    if not fig:
        figs = list(map(plt.figure, plt.get_fignums()))
    else:
        figs=[fig]
    
    for fig in figs:
        for axi in fig.axes:
            axi.set_rasterization_zorder(1)
            for linei in axi.lines:
                linei.set_zorder(linei.get_zorder()-100)


def generate_subplot_grid(grid=(2,1),figsize=None,share_y=False,share_x=False
                         ,hspace=False,wspace=False,right=0.97,left=0.1,top=0.95,bottom=0.12
                         ,twinaxx=False,twinaxy=False,twincolor=['orange','blue']
                         ,labelpanels=False,flatten_axes=True,force_flatten_axes=False,add_cbar=False):
    fig=plt.figure(figsize=figsize)
    
    axes=[[0]*grid[1] for _ in range(grid[0])]
    if twinaxx:
        axes_twin=[[0]*grid[1] for _ in range(grid[0])]
        right-=0.07
        #left-=0.04
        
    i=0
    
    if add_cbar:
        xfactor=20
        fgrid=(grid[0],grid[1]*(xfactor+1))
    else:
        fgrid=grid
        xfactor=1
    
    for iy in range(grid[0]):
        for ix in range(grid[1]):
            axes[iy][ix] = plt.subplot2grid(fgrid,(iy,ix*xfactor),rowspan=1,colspan=xfactor)
            
            if twinaxx:
                axes_twin[iy][ix]=axes[iy][ix].twinx()
                axes[iy][ix].tick_params(axis='y', colors=twincolor[0])    #setting up X-axis tick color to red
                axes_twin[iy][ix].tick_params(axis='y', colors=twincolor[1])
                if labelpanels:
                    for twini in [0,1]:
                        axes[iy][ix].text(0.4,0.5,f'{twini},',horizontalalignment='center'
                                         ,verticalalignment='center'
                                         ,transform=axes[iy][ix].transAxes
                                         ,color=twincolor[twini])
            
            if share_x and iy!=grid[0]-1:
                axes[iy][ix].tick_params(labelbottom=False)
                hspace=0
                # if twinaxy:
                #     axes_twin[iy][ix].tick_params(labelbottom=False)
                    
            if share_y:
                if ix!=0:
                    axes[iy][ix].tick_params(labelleft=False)
                    if not wspace:
                        wspace=0    
                if twinaxx:
                    if ix!=grid[1]-1:
                        axes_twin[iy][ix].tick_params(labelright=False)
                    if ix!=0:
                        axes[iy][ix].tick_params(labelleft=False,left=False)
                        
            if labelpanels:
                axes[iy][ix].text(0.5,0.5,f'{iy}, {ix}',horizontalalignment='center'
                                 ,verticalalignment='center'
                                 ,transform=axes[iy][ix].transAxes)
                
            #axes[iy][ix].set_title(f'{ix}  {iy}')
            
            if share_x and iy>0:
                axes[iy][ix].sharex(axes[iy-1][ix])
            if share_y and ix>0 :
                axes[iy][ix].sharey(axes[iy][ix-1])
                if twinaxx:
                    axes_twin[iy][ix].sharey(axes_twin[iy][ix-1])
            i+=1
                   
    if isinstance(hspace,bool):
        hspace = 0.15*grid[0]
    
    if isinstance(wspace,bool):
        wspace = 0.15*grid[1]
                
    if (grid[1]==1 or grid[0]==1) and flatten_axes or force_flatten_axes:
        axes= [x for xs in axes for x in xs]
        if twinaxx:
            axes_twin= [x for xs in axes_twin for x in xs]
            
        if len(axes)==1:
            axes=axes[0]
            
    if add_cbar:
        cbar=plt.subplot2grid(fgrid,(0,fgrid[1]-grid[0]+2),rowspan=grid[0],colspan=grid[0])
        right=right-0.02
        
    fig.subplots_adjust(bottom=bottom,top=top,left=left,right=right,hspace=hspace,wspace=wspace)
        
    if add_cbar:
        if twinaxx:
            return axes,axes_twin,cbar
        return axes,cbar
    
    if twinaxx:
        return axes,axes_twin
    return axes

def update_colors_for_ploting(data,colors,scale='linear',return_norm=False):
    
    if isinstance(colors,str):
        cmap = plt.colormaps[colors]
        colors=cmap
    
    if not isinstance(colors, (list, np.ndarray,c.LinearSegmentedColormap,c.ListedColormap)): 
        if not colors:
            colors=None
        colors=[colors]*len(data)
        return None,colors
    
    if not isinstance(colors,(c.LinearSegmentedColormap,c.ListedColormap)) and len(colors)<len(data):
        cmap = c.LinearSegmentedColormap.from_list("", colors) 
        colors=cmap
    else:
        cmap = colors
    
    if isinstance(colors,(c.LinearSegmentedColormap,c.ListedColormap)):
        colorss=[0]*len(data)
        
        if scale == 'linear':
            norm = c.Normalize(vmin=np.min(data), vmax=np.max(data))
        elif scale == 'log':
            norm = c.LogNorm(vmin=np.min(data), vmax=np.max(data))
    
    for i_c in range(len(data)):
        colorss[i_c] = cmap(norm(data[i_c]))
        
        colors=colorss

    if return_norm:
        return cmap,colors,norm
    return cmap,colors

def cut_cmap(cmapname='viridis',minmax=[0,1],cmap=False):
    from matplotlib.colors import ListedColormap
    
    if isinstance( cmap,bool):
        cmap = plt.colormaps[cmapname]
        
    cmap=cmap.resampled(256)
    newcolors = cmap(np.linspace(minmax[0],minmax[1], 256))
    # transparent = np.array([256/256, 256/256, 256/256, 0])
    # blue=np.array([0/256, 0/256, 256/256, 1])
    # newcolors[:12, :] = blue
    # newcolors[12:28, :] = transparent
    # cmap = ListedColormap(newcolors)
    
    return ListedColormap(newcolors)

def make_cmap(colors=None):
    
    if colors is None:
        colors = [
            (149, 35, 76),
            (199, 82, 91),
            (229, 123, 87),
            (236, 177, 114),
            (82 , 140, 188),
            (100, 89 ,162)
            ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    cmap = c.LinearSegmentedColormap.from_list('my_cmap', normalized_colors, N=256)

    return cmap


def avoid_edge_ticks(ax, axis='both', frac=0.05):
    
    if axis in ('x', 'both'):
        axmin, axmax = ax.get_xlim()
        span=axmax-axmin
        ticks=ax.get_xticks()
        
        if any(np.abs(ticks-axmin)<frac*span) or any(np.abs(ticks-axmax)<frac*span):
            ax.set_xlim((axmin+frac*span,axmax-frac*span))
            ticks=ax.get_xticks()
            ticks=ticks[(ticks>axmin+frac*span) & (ticks<axmax-frac*span) ]
            
            ax.set_xticks(ticks)
            ax.set_xlim(axmin,axmax)
            

    if axis in ('y', 'both'):
        axmin, axmax = ax.get_ylim()
        span=axmax-axmin
        ticks=ax.get_yticks()
        
        if any(np.abs(ticks-axmin)<frac*span) or any(np.abs(ticks-axmax)<frac*span):
            ax.set_ylim((axmin+frac*span,axmax-frac*span))
            ticks=ax.get_yticks()
            ticks=ticks[(ticks>axmin+frac*span) & (ticks<axmax-frac*span) ]
            
            ax.set_yticks(ticks)
            ax.set_ylim(axmin,axmax)
            
def get_log_edges_for_pcolormesh(centers):
    log_c = np.log10(centers)
    diff_log = np.diff(log_c)
    
    log_edges = np.zeros(len(centers) + 1)
    log_edges[1:-1] = log_c[:-1] + diff_log / 2
    log_edges[0] = log_c[0] - diff_log[0] / 2
    log_edges[-1] = log_c[-1] + diff_log[-1] / 2
    return 10**log_edges
            
