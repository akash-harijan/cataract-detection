from model import create_model
from ..data import DataLoader
from pathlib import Path

import os




if __name__ == '__main__':

    model = create_model()

    project_dir = Path(__file__).resolve().parents[2]
    print("Project dir {0}".format(project_dir))

    loader = DataLoader(os.path.join(project_dir, "data/external"))
    x_train, x_test, y_train, y_test = loader.load_data()
