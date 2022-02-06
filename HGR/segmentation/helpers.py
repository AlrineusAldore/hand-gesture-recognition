from inspect import currentframe
import matplotlib.pyplot as plt

def get_line_num():
    return "line num: " + str(currentframe().f_back.f_lineno)

def start_segmentation_plot(hist_h, hist_s, hist_v):
    """
    :param hist_h: hue historgram
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
