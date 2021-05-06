import json
import os

from vcd.application.main_logic import VIDEO_PATH_KEY, REGRESSION_MODEL_PATH_KEY


def read_config_file(path):
    with open(path) as f:
        return json.load(f)


def create_config_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


if __name__ == '__main__':
    # video_path = input("Full path to video:")

    config_path = os.path.join(os.getcwd(), "vcd/resources/config/config.json")

    # config_data = {
    #     VIDEO_PATH_KEY: "C:/Users/ahumyck/PycharmProjects/diplom/vcd/resources/video/result.mp4",
    #     ALGORITHM_TYPE_KEY: AlgorithmType.MSE,
    #     REGRESSION_MODEL_PATH_KEY: ""
    # }
    #
    # create_config_file(config_data, config_path)

    config = read_config_file(config_path)
    video_path = config[VIDEO_PATH_KEY]
    video_name = os.path.basename(video_path)
    folder_path = os.path.dirname(video_path)
    template_name = os.path.splitext(video_name)[0]

    print("folder_path:", folder_path)
    print("template_name:", template_name)
    print("video_name:", video_name)
