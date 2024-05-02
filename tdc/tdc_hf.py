from huggingface_hub import create_repo
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import os

import torch
from .ml_submodules import _MODELS, _CALLBACKS 

deeppurpose_repo = [
    'hERG_Karim-Morgan',
    'hERG_Karim-CNN',
    'hERG_Karim-AttentiveFP',
    'BBB_Martins-AttentiveFP',
    'BBB_Martins-Morgan',
    'BBB_Martins-CNN',
    'CYP3A4_Veith-Morgan',
    'CYP3A4_Veith-CNN',
    'CYP3A4_Veith-AttentiveFP',
]

hf_repo_torch = [
    "mli-PINNACLE",
]


class tdc_hf_interface:
    '''
    Example use cases:
    # initialize an interface object with HF repo name
    tdc_hf_herg = tdc_hf_interface("hERG_Karim-Morgan")
    # upload folder/files to this repo
    tdc_hf_herg.upload('./Morgan_herg_karim_optimal')
    # load deeppurpose model from this repo
    dp_model = tdc_hf_herg.load_deeppurpose('./data')
    dp_model.predict(XXX)
    '''

    def __init__(self, repo_name):
        self.repo_id = "tdc/" + repo_name
        self.model_name = repo_name.split('-')[1]

    def upload(self, folder_path):
        create_repo(repo_id=self.repo_id)
        api = HfApi()
        api.upload_folder(folder_path=folder_path,
                          path_in_repo="model",
                          repo_id=self.repo_id,
                          repo_type="model")

    def file_download(self, save_path, filename):
        model_ckpt = hf_hub_download(repo_id=self.repo_id,
                                     filename=filename,
                                     cache_dir=save_path)

    def torch_file_download(self, save_path, filename):
        return hf_hub_download(repo_id=self.repo_id,
                                     filename=filename,
                                     cache_dir=save_path)

    def repo_download(self, save_path):
        snapshot_download(repo_id=self.repo_id, cache_dir=save_path)
        
    def load(self, save_path="./tmp", has_data_dir=True):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if self.repo_id[4:] in hf_repo_torch:
            save_path = save_path + '/' + self.repo_id[4:]
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # model = self.torch_file_download(save_path, "model/model.pt")
            # ckpt = self.torch_file_download(save_path, "model/config.ckpt")
            # get the right model class
            model_class = _MODELS.get(self.model_name)
            assert model_class is not None, f"{self.model_name} is not configured with an appropriate model class."
            self.repo_download(save_path)
            # data_path = str(save_path)

            # save_path = save_path + '/models--tdc--' + self.repo_id[
            #     4:] + '/blobs/'

            save_path = save_path + '/models--tdc--' + self.repo_id[4:] + "/"
            
            # get model_file
            model_file = None
            # for f in os.listdir(save_path):
            #     if f.endswith(".pt"):
            #         model_file = f
            #         break
            for root, dirs, files in os.walk(save_path):
                for file in files:
                    if file.endswith(".pt"):
                        model_file = os.path.join(root, file)
                        break
            assert model_file is not None, "No model file found in {}".format(save_path)
            os.rename(model_file, save_path+"model.pt")
            # get config_file
            # file_name1 = save_path + os.listdir(save_path)[0]
            # file_name2 = save_path + os.listdir(save_path)[1]

            # if os.path.getsize(file_name1) > os.path.getsize(file_name2):
            #     model_file, config_file = file_name1, file_name2
            # else:
            #     config_file, model_file = file_name1, file_name2

            # os.rename(model_file, save_path + 'model.pt')
            # os.rename(config_file, save_path + 'config.ckpt')

            # if has_data_dir:
            #     api = HfApi()
            #     repo_files = api.list_repo_files(self.repo_id[4:])
            #     for file in repo_files:
            #         if file.startswith("data"):
            #             self.torch_file_download(data_path, file)
            
            cb = _CALLBACKS.get(self.model_name)
            assert cb is not None, "{} does not have configured call back".format(self.model_name)
            net = cb(model_class, save_path)
            net.load_state_dict(torch.load(model_file))
            return net
        else:
            raise ValueError("This repo not in configured list of pytorch model repos {}".format(",".join(hf_repo_torch)))

    def load_deeppurpose(self, save_path):
        if self.repo_id[4:] in deeppurpose_repo:
            save_path = save_path + '/' + self.repo_id[4:]
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.file_download(save_path, "model/model.pt")
            self.file_download(save_path, "model/config.pkl")

            save_path = save_path + '/models--tdc--' + self.repo_id[
                4:] + '/blobs/'
            file_name1 = save_path + os.listdir(save_path)[0]
            file_name2 = save_path + os.listdir(save_path)[1]

            if os.path.getsize(file_name1) > os.path.getsize(file_name2):
                model_file, config_file = file_name1, file_name2
            else:
                config_file, model_file = file_name1, file_name2

            os.rename(model_file, save_path + 'model.pt')
            os.rename(config_file, save_path + 'config.pkl')
            try:
                from DeepPurpose import CompoundPred
            except:
                raise ValueError(
                    "Please install DeepPurpose package following https://github.com/kexinhuang12345/DeepPurpose#installation"
                )

            net = CompoundPred.model_pretrained(path_dir=save_path)
            return net
        else:
            raise ValueError("This repo does not host a DeepPurpose model!.. try calling load() for a standard model")

    def predict_deeppurpose(self, model, drugs):
        try:
            from DeepPurpose import utils
        except:
            raise ValueError(
                "Please install DeepPurpose package following https://github.com/kexinhuang12345/DeepPurpose#installation"
            )
        if self.model_name == 'AttentiveFP':
            self.model_name = 'DGL_' + self.model_name
        X_pred = utils.data_process(X_drug=drugs,
                                    y=[0] * len(drugs),
                                    drug_encoding=self.model_name,
                                    split_method='no_split')
        y_pred = model.predict(X_pred)[0]
        return y_pred
