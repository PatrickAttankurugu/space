import os
import gdown

def download_weights():
    weights_dir = 'ml_models/deepfake'
    os.makedirs(weights_dir, exist_ok=True)
    
    weights = {
        'final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36': 'https://drive.google.com/uc?id=1Q8EDSx1jOFx4SGv90YkEVeVnksADjHcm',
        'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19': 'https://drive.google.com/uc?id=1ypnKmX7NvNfo6RYcOWZehEDQEHQScs1O',
        'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31': 'https://drive.google.com/uc?id=1M_VRMvLjC3WLgMjH9eIszC5x7wbSG1YR'
    }

    for name, url in weights.items():
        output = os.path.join(weights_dir, name)
        if not os.path.exists(output):
            print(f'Downloading {name}...')
            gdown.download(url, output, quiet=False)

if __name__ == "__main__":
    download_weights()