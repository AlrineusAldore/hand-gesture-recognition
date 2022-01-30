from enum import Enum
#import screen_brightness_control as sbc

class CommandsHandler:
    def __init__(self):
        self.mouse_controlled = False

    def sign_to_command(self, sign):
        pass


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


def brightness():
    # get current brightness value
    #print(sbc.get_brightness())
    pass
