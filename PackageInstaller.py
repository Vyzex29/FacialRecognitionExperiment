import subprocess
import sys
import Configuration.Config as config
import os
from git import Repo #https://gitpython.readthedocs.io/en/stable/tutorial.html

dependencies =["gitpython", "opencv-python", "pyqt5", "labelimg", "wget", "tensorflow-gpu", "protobuf", "matplotlib"]
gitRepos = [{"repo": "https://github.com/tzutalin/labelImg", "path": config.paths["LABELIMG_PATH"]},
            {"repo": "https://github.com/tensorflow/models", "path": config.paths["APIMODEL_PATH"]},
            {"repo": "https://github.com/Vyzex29/GenerateTFRecord", "path": config.paths["SCRIPTS_PATH"]}
            ]
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def installDependencies():
    for path in config.paths.values():
        if not os.path.exists(path):
            print("Created directory at: " + path)
            os.mkdir(path)

    for dependency in dependencies:
        install(dependency)

    for gitRepo in gitRepos:
        try:
            repo = Repo.clone_from(gitRepo["repo"], gitRepo["path"])
        except:
            print(gitRepo)
            print("Already cloned")


