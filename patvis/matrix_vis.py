# utilizites for plotting a matrix. 
# Currently supported is a heatmap and 3d histogram. 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot3d(matrix, show=True, savefn=None, plt_type='bar', 
           xlabel=None, ylabel=None, zlabel=None, title=None):
    """
    Plot the input matrix in 3d. Either save, show, or both.
    Supported types are 'bar', 'tri', and 'surface'. 
    """
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111, projection='3d')
    x_data,y_data = np.meshgrid(np.arange(matrix.shape[1]), 
                                np.arange(matrix.shape[0]))
    
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = matrix.flatten()
    if plt_type=='bar':
        ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    elif plt_type=='tri':
        ax.plot_trisurf(x_data, y_data,z_data,cmap=cm.jet, linewidth=.2)
    elif plt_type=='surface':
        pass
    else:
        raise RuntimeError('Plot type {} not supported.'.format(plt_type))
    if xlabel is not None:
        plt.ylabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)
    if zlabel is not None:
        plt.zlabel(zlabel)
    if title is not None:
        plt.title(title)
    if savefn is not None:
        plt.savefig(savefn, dpi=100)
    if show:
        plt.show()
    return fig

def heatmap(matrix, show=True,savefn =None,
            xlabel=None,ylabel=None,title=None):
    mx = round(np.max(matrix),1)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.imshow(matrix.transpose(),origin='lower',aspect='auto')
    cbar = plt.colorbar(ticks=[0,mx])
    cbar.ax.set_yticklabels([0,mx])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if savefn is not None:
        plt.savefig(savefn, dpi=100)
    if show:
        plt.show()
    return fig
    

    
