# Constants and Enums for the commands handler
from enum import Enum


class Sign(Enum):
    index_only = "move_mouse"
    index_middle_closed = "double_click"
    index_middle_ring_closed = "right_click"
    like = "increase_brightness"
    dislike = "decrease_brightness"
    all_fingers_open = "mute"
    ok_sign = "got_you"
    index_middle_open = "increase_volume"
    index_middle_ring_open = "decrease_volume"


class Commands(Enum):
    move_mouse = "index_only"
    double_click = "index_middle_closed"
    right_click = "index_middle_ring_closed"
    increase_brightness = "like"
    decrease_brightness = "dislike"
    mute = "all_fingers_open"
    got_you = "ok_sign"
    increase_volume = "index_middle_open"
    decrease_volume = "index_middle_ring_open"
