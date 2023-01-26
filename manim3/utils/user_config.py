"""
Description
===========
This py file is used to receive user's configuration, and store them into a JSON config file.
"""

import os
import sys
import json


def os_check() -> str:
    """
    Description
    ===========
    Check the OS of the user.
    """
    match sys.platform:
        case "win32":
            return "Windows"
        case "linux":
            return "Linux"
        case "darwin":
            return "MacOS"
        case _:
            return "Unknown"


def file_field_set() -> list[bool, str]:
    """
    Description
    ===========
    Config file has a certain field that it can affect on.
    Therefore, this function is to make sure the correct field is chosen by user.

    Return
    ======
    A list, two parts:
    First part: bool, is this step of config successful?
    Second part: str, if successful, the FULL path to the config file,
                      if failed. None.
    """
    print("""
====== You now @ File Field Settings. ======
A config file has two fields: "global" and "local".
    # local: Generate a config file in the current directory. Have higher priority than global.
    # global: Generate a config file in your home directory, affect all projects that not have a local config file.
    # If you want to generate a local config file, please input "local", otherwise, input "global".
Default is: "local", Use an ENTER key to quickly choice it.""")
    field = input()
    if field == "":
        field = "local"
    if field not in ["local", "global"]:
        print("Invalid input. This setting will be restarted soon.")
        return [False, None]
    path = os.path.join(os.getcwd(), "manim3.json") if field == "local" \
        else os.path.join(os.path.expanduser("~"), "manim3.json")
    print(f"Your config file will be saved in {path}.")
    return [True, path]


def tex_temp_dir_set() -> list[bool, dict]:
    """
    Description
    ===========
    LaTeX will generate middle files, so set dir to store them.

    Return
    ======
    A list, two parts:
    First part: bool, is this step of config successful?
    Second part: dict, if successful, {"tex_temp_dir": <path_to_dir>},
                       if failed. None.
    """
    print("""
====== You now @ LaTeX Temporary Directory Settings. ======
When you use LaTeX, a temporary directory will be needed in order to store the generated files like .tex or .dvi.
    # If you want to set dir manually, please input the FULL path to your temp dir, otherwise
    # use SYSTEM Temp dir.
Default is: SYSTEM Temp dir, Use an ENTER key to quickly choice it.""")
    path = input()
    if path != "":  # Manually set temp dir
        if not os.path.exists(path):
            print("Directory do not exist. This setting will be restarted soon.")
            return [False, None]
        print(f"Your LaTeX temporary directory will be set to {path}.")
        return [True, {"tex_temp_dir": path}]
    else:  # Use system temp dir
        print("Your LaTeX temporary directory will be set to SYSTEM Temp dir.")
        OS = os_check()
        match OS:
            case "Windows":
                return [True, {"tex_temp_dir": os.getenv("TEMP")}]
            case "Linux" | "MacOS":
                return [True, {"tex_temp_dir": "/tmp"}]
            case "Unknown":
                print("Unknown OS. Please manually input temp path, This setting will be restarted soon.")
                return [False, None]


def run_settings() -> None:
    """
    Description
    ===========
    Main function of this py file.

    Return
    ======
    None
    """

    # setting-items number
    SETTING_ITEM_NUMS = 2

    # make them all False to initialize
    results = [
        [False, None] for _ in range(SETTING_ITEM_NUMS)
    ]
    is_successful = all(results[i][0] for i in range(SETTING_ITEM_NUMS))

    # run settings
    while not is_successful:
        # In first time, all results are False, so all of them will be run.
        # After that, only the False item will be run.
        if not results[0][0]:
            results[0] = file_field_set()
        if not results[1][0]:
            results[1] = tex_temp_dir_set()

        # check if all settings are successful
        is_successful = all(results[i][0] for i in range(SETTING_ITEM_NUMS))

    # all settings process are successful, now write them into a JSON file.
    json_file_path = results[0][1]
    json_file_content = {}
    for i in range(1, SETTING_ITEM_NUMS):
        json_file_content.update(results[i][1])

    with open(json_file_path, "w") as f:
        json.dump(json_file_content, f, indent=4)

    print("Config file generated successfully.")
