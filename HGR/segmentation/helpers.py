from inspect import currentframe
import matplotlib.pyplot as plt


def get_line_num():
    return "line num: " + str(currentframe().f_back.f_lineno)


def start_segmentation_plot(hist_h, hist_s, hist_v):
    """
    :param hist_h: hue histogram
    :param hist_s: saturation histogram
    :param hist_v: value histogram
    :return: None
    """
    plt.cla()
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")


def end_segmentation_plot(h_range, s_range, v_range):
    """
    :param h_range: tuple consisting the start and end of hue
    :param s_range: tuple consisting the start and end of saturation
    :param v_range: tuple consisting the start and end of value
    :types: all params are tuples consisting of 2 ints
    :return: None
    """
    plt.title(f"hMin:{h_range[0]}, hMax:{h_range[1]}, sMin:{s_range[0]}, sMax:{s_range[1]}, vMin:{v_range[0]}, vMax:{v_range[1]}")

    plt.legend()
    plt.pause(0.001)


# Gets 2 points and returns the slope between them
def slope(pt1, pt2, f=None):
    """
    :param pt1: or (x1, y1) or just x1, depending on f
    :param pt2: or (x2, y2) or just x2, depending on f
    :param f: math function f(x), if None then pts are (x,y), if f is given then pts are just x
    :return: The slope between pt1 and pt2
    """
    if f is None:
        return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    else:
        return (f[pt2]-f[pt1])/(pt2-pt1)
