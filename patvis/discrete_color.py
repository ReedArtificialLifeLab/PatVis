# Generate a color scheme by equal spacing along the color wheel.
import numpy as np
import matplotlib.pyplot as plt

def _test_scheme(colors,show=True,savefn=None):
    """
    Tests a color scheme (list of colors) by plotting equally spaced in a row,
    one for each color.
    """
    fig = plt.figure()
    fig.set_size_inches(18.5,10.5)
    n = len(colors)
    xs = np.arange(n)
    ys = np.array([1 for _ in range(n)])
    plt.scatter(xs,ys,color=colors)
    if savefn is not None:
        plt.savefig(savefn,dpi=100)
    if show:
        plt.show()

def _hsl_to_rgb(hsl):
    h,s,l = hsl
    rgb = [l,l,l]
    if s != 0:
        chroma = 1 - np.abs(2*l-1)
        hp = h/60.
        x = chroma*(1 - np.abs(hp%2 - 1))
        m = l - chroma/2.
        if 0 <= hp < 1:
            rgb = [chroma+m, x+m, m]
        elif 1 < hp < 2:
            rgb = [x+m, chroma+m, m]
        elif 2 < hp < 3:
            rgb = [m, chroma+m, x+m]
        elif 3 < hp < 4:
            rgb = [m, x+m, chroma+m]
        elif 4 < hp < 5:
            rgb = [x+m, m, chroma+m]
        elif 5 < hp < 6:
            rgb = [chroma+m, m, x+m]
        else:
            raise RuntimeError("{} is an invalid HSL color.".format((h,s,l)))
    return rgb


def discrete_color_scheme(n=10,home_color=(193,60,35)):
#def discrete_color_scheme(n=10,home_color=(193,95,25)): # I wanted to make the default darker. Still cyan don't worry.
    """
    Generate a color scheme based on n colors evenly space around the color wheel,
    in terms of angle. The home_color is the default starting point, which is
    cyan, in hsl notation. We can divide the 2(pi) radians by n to get the angle offset.
    Then the angles are given by (start_angle) + (2*pi*i)/n as i varies from 0 to n-1.
    """
    angle_offset = 360./n
    home_angle, sat, lum = home_color
    angles = [(home_angle+i*angle_offset)%360 for i in range(n)]
    hsls = [(ang,sat/100.,lum/100.) for ang in angles]
    rgbs = [_hsl_to_rgb(hsl) for hsl in hsls]
    return rgbs

def rgb_to_hex(rgb):
    r,g,b = [int(x*255) for x in rgb]
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

if __name__ == '__main__':
    colors = discrete_color_scheme(6)
    _test_scheme(colors, savefn = None,show=True)
