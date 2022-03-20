from constants import *
import matplotlib.pyplot as plt
import scipy.signal as signal
import cv2


def start_segmentation_plot(hist1, hist2, hist3, colors_space):
    """
    :param hist1: first channel histogram
    :param hist2: second channel histogram
    :param hist3: third channel histogram
    :param colors_space: name of the color space
    :return: None
    """
    plt.cla()
    plt.plot(hist1, color='r', label=colors_space[0])
    #plt.plot(hist2, color='g', label=colors_space[1])
    #plt.plot(hist3, color='b', label=colors_space[2])


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



def check_for_endpoints_extrema(f):
    """
    Checks if endpoints of f(x) are significant min/max and return them if they are
    :param f: Math function f(x)
    :return: significant min/max endpoints in the form of (mins=[], maxes=[])
    """
    maxes = []
    mins = []

    #Get slopes of endpoints with points that has x differences of 3
    start_slope = slope(0, 3, f=f)
    end_slope = slope(len(f)-4, len(f)-1, f=f)

    # Check if start is a significant min/max
    if start_slope > SMALL_SLOPE*2:
        mins.append(0)
    elif start_slope < SMALL_SLOPE*(-2):
        maxes.append(0)

    #Check if end is a significant min/max
    if end_slope > SMALL_SLOPE*2:
        maxes.append(len(f)-1)
    elif end_slope < SMALL_SLOPE*(-2):
        mins.append(len(f)-1)

    return mins, maxes



def check_for_value(f, value, start=0, end=256, go_backwards=False):
    """
    Function checks the first occurrence of value in function f(x) within the given range and returns it
    :param f: math function f(x) with x between 0 and len(f)-1
    :param value: wanted y value in function
    :param start: from when should we start
    :param end: when should we stop
    :param go_backwards: Whether to check the first from the start or first from the end
    :return: the first x of the wanted y value
    """
    # Can't initialize it so from the start so set it here instead
    if end == 256:
        end = len(f)

    if go_backwards:
        res = start
        for x in range(start, end):
            if int(f[end-x]) == value:
                res = end-x
                break
    else:
        res = end
        for x in range(start, end):
            if int(f[x]) == value:
                res = x
                break

    return res




# Gets the range of a max point (left of it to right of it)
def get_range_of_max(f, max, pts, max_pts):
    try:
        max_i = pts.index(max)
    except:
        try:
            # If max is from original maxima list and not in the processed list then take closest maxima of it
            max = min(max_pts, key=lambda x:abs(x-max))
            max_i = pts.index(max)
        except Exception as e:
            print("e:", repr(e))
            print("max_pts:", max_pts)
            raise e


    # Get min point / zero point to the left of max (whichever is closest to max)
    start = check_for_value(f, 0, end=max, go_backwards=True)
    if max_i != 0:
        left_min = pts[max_i - 1]
        if left_min > start:
            start = left_min

    # Get min point / zero point to the right of max (whichever is closest to max)
    end = check_for_value(f, 0, start=max)
    if max_i != len(pts) - 1:
        right_min = pts[max_i + 1]
        if right_min < end:
            end = right_min

    return start, end


# Get range of max hill (max point until zero from left & right)
def get_range_of_hill_range(f, max):
    start = check_for_value(f, 0, end=max, go_backwards=True)
    end = check_for_value(f, 0, start=max)

    return start, end



# Remove any unnecessary extreme points with similar values
def get_useful_extrema(f):
    # All local min & max points
    minima = signal.argrelmin(f)[0].tolist()
    maxima = signal.argrelmax(f)[0].tolist()

    pts = sorted(minima+maxima)
    pts_copy = pts.copy() #for debugging purposes

    # Go through all pts except first and last
    for i in range(1, len(pts)-1):
        # If extreme points are really close to each other
        if pts[i] - pts[i-1] < 5 and pts[i+1] - pts[i] < 5:
            # And if the slope between them is small enough
            if abs(slope(pts[i-1], pts[i], f=f)) < SMALL_SLOPE and abs(slope(pts[i], pts[i+1], f=f)) < SMALL_SLOPE:
                # Get rid of appropriate min/max depending on current point
                if pts[i] in minima:
                    a = minima
                    b = maxima
                elif pts[i] in maxima:
                    a = maxima
                    b = minima
                else:
                    continue

                a.remove(pts[i])  # Remove useless A point
                # Replace 2 B points with 1 B point in the middle
                mid = ((pts[i-1]+pts[i+1]))//2
                try:
                    b.remove(pts[i+1])
                except Exception as e:
                    print("\nbreakpoint this exception. i =", i, ". pt=", pts_copy[i], pts[i])
                    print("e:", repr(e))
                    print("pts at the start:",pts_copy)
                    print("pts at the error:",pts)
                b[:] = [mid if x==pts[i-1] else x for x in b]  # Change pts[i-1]'s value to mid

    # If endpoints are found to be significant min/max, then add them to minima/maxima accordingly
    end_mins, end_maxes = check_for_endpoints_extrema(f)

    for x in end_mins:
        index = 0
        if x == len(f)-1:
            index = len(minima)
        minima.insert(index, x)

    for x in end_maxes:
        index = 0
        if x == len(f)-1:
            index = len(minima)
        minima.insert(index, x)


    return minima, maxima


def check_all_zero(iterable):
  return all(v == 0 for v in iterable)



def open_close(img, bin_img, kernel_size=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    closing_mask = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel)
    opening_mask_res = cv2.bitwise_and(img, img, mask=opening_mask)

    return opening_mask, opening_mask_res
