import shutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import utils as fcs

root = fcs.get_parent_path(lvl=1)
path_des = os.path.join(root, 'src', 'data', 'pretrain_model')

def main(model):
    path_source = os.path.join(root, 'data', 'offline_training', model)
    path_model = os.path.join(path_source, 'checkpoint_epoch_5000.pth')
    path_params = os.path.join(path_source, 'src', 'data', 'pretraining', 'norm_params')

    move_and_rename_file(path_model, os.path.join(path_des, 'model.pth'))
    move_and_rename_file(path_params, os.path.join(path_des, 'norm_params'))

def move_and_rename_file(source, destination):
    """
    move and rename the model
    """
    try:
        if not os.path.isfile(source):
            print(f"Error: The source file '{source}' does not exist.")
            return
    
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        shutil.copy2(source, destination)
        print(f"File '{source}' successfully moved and renamed to '{destination}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    model = 'small_pro'
    main(model)

