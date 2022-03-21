import sys
import json


class Settings(object):

    __slots__ = (
                 "dataset_dir",
                 "batch_size",
                 "epochs",
                 "learning_rate",
                 "image_size",
                 "gpus_ids",
                 )

    def __init__(self):

        try:
            settings_config = sys.argv[1]
            user_settings = json.load(open(settings_config, "r"))
            self.get_settings(user_settings)

        except IndexError as exc:
            print("No config found. Exception", exc)
            exit(-1)
        except FileNotFoundError as exc:
            print("Config argument found, but is not a file. Exception", exc)
            exit(-1)
        except AttributeError as exc:
            print("Wrong setting provided in the file. Exception", exc)
            exit(-1)

    def get_settings(self, json_opened):
        for key in json_opened:
            setattr(self, key, json_opened[key])
        sett = list((getattr(self, attr) for attr in self.__slots__))
        return sett
