from huggingface_hub import create_repo
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
import os

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

model_hub = ["Geneformer", "scGPT"]


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
        try:
            self.model_name = repo_name.split('-')[1]
        except:
            self.model_name = repo_name

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

    def repo_download(self, save_path):
        snapshot_download(repo_id=self.repo_id, cache_dir=save_path)

    def load(self):
        if self.model_name not in model_hub:
            raise Exception("this model is not in the TDC model hub GH repo.")
        elif self.model_name == "Geneformer":
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained("tdc/Geneformer")
            return model
        elif self.model_name == "scGPT":
            from transformers import AutoConfig, AutoModel
            from .models.scgpt import ScGPTModel, ScGPTConfig
            AutoConfig.register("scgpt", ScGPTConfig)
            AutoModel.register(ScGPTConfig, ScGPTModel)
            model = AutoModel.from_pretrained("tdc/scGPT")
            return model
        raise Exception("Not implemented yet!")

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
            raise ValueError("This repo does not host a DeepPurpose model!")

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
