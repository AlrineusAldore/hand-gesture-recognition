from commands.constants import *
<<<<<<< HEAD
=======
from info_handlers.data import Data
>>>>>>> develop
import screen_brightness_control as sbc


class CommandsHandler:
<<<<<<< HEAD
    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data
=======
    def __init__(self, data=Data()):
        self.data = data
>>>>>>> develop

        self.mouse_controlled = False


    def update_data(self, data):
        self.data = data


    #  Do appropriate commands based on how many fingers are up
    def count_fingers(self):
<<<<<<< HEAD
        cnt = None
        if "fings_count" in self.data:
            cnt = self.data["fings_count"]
=======
        cnt = self.data.fings_count
>>>>>>> develop

        #  Do nothing if there are no fingers up
        if cnt is None or cnt == 0:
            return None

        #  Change brightness for 4 and 5 fingers
        if cnt == 5:
            change_brightness(10)
            return Commands.increase_brightness
        elif cnt == 4:
            change_brightness(-10)
            return Commands.decrease_brightness


    def check_commands(self):
        result = self.count_fingers()

        if result is None:
            return None

        if type(result) is Commands:
            result = Sign[result.value]

        return result.value


    def sign_to_command(self, sign):
        pass






#  changes the brightness by val
def change_brightness(val):
    b = sbc.get_brightness(display=0)
    sbc.set_brightness(b+val, display=0)
