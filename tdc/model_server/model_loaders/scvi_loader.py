class scVILoader():

    def __init__(self):
        pass

    def load(self, census_version):
        import requests
        import os

        scvi_url = f"https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/{census_version}/homo_sapiens/model.pt"
        os.makedirs(os.path.join(os.getcwd(), 'scvi_model'), exist_ok=True)

        output_path = os.path.join('scvi_model', 'model.pt')

        try:
            response = requests.get(scvi_url, verify=False)
            if response.status_code == 404:
                raise Exception(
                    'Census version not found, defaulting to version 2024-07-01'
                )
        except Exception as e:
            print(e)
            census_version = "2024-07-01"
            scvi_url = f"https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/2024-07-01/homo_sapiens/model.pt"
            response = requests.get(scvi_url, verify=False)

        with open(output_path, "wb") as file:
            file.write(response.content)

        print(
            f'scVI version {census_version} downloaded to {output_path} in current directory'
        )
