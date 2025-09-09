"""
O seguinte é um exemplo de algoritmo.

Ele foi feito para rodar dentro de um contêiner.

Para rodar o contêiner localmente, você pode chamar o seguinte script bash:

  ./do_test_run.sh

Isso irá iniciar a inferência e ler de ./test/input e escrever para ./test/output

Para salvar o contêiner e prepará-lo para upload para Grand-Challenge.org você pode chamar:

  ./do_save.sh

Qualquer contêiner que demonstre o mesmo comportamento servirá, este é puramente um exemplo de como alguém PODE fazê-lo.

Consulte a documentação para obter detalhes sobre o ambiente de execução na plataforma:
https://grand-challenge.org/documentation/runtime-environment/

Feliz programação!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch # Importe o PyTorch

# Importa a classe do seu modelo Swin
from model.swin_model import SwinTinyClassificationModel

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    interface_key = get_interface_key()

    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]

    return handler()


def interface_0_handler():
    input_stacked_barretts_esophagus_endoscopy_images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    
    _show_torch_cuda_info()
    
    print('Swin-Tiny-RARE25')
    model = SwinTinyClassificationModel(
        weights=RESOURCE_PATH / "best_swin_tiny_rare25.pth",
        num_classes=2,
    )
    
    output_stacked_neoplastic_lesion_likelihoods = model.predict(input_stacked_barretts_esophagus_endoscopy_images)

    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=output_stacked_neoplastic_lesion_likelihoods,
    )

    return 0


def get_interface_key():
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    # A sua lógica de inferência espera uma lista de imagens. 
    # Precisamos tratar o retorno do SimpleITK, que geralmente é um array 3D.
    result = SimpleITK.ReadImage(input_files[0])
    img_array = SimpleITK.GetArrayFromImage(result)
    
    # Se a imagem for 3D (C, H, W) ou (H, W, C), o SimpleITK.GetArrayFromImage() irá retornar
    # um array 3D. A sua função 'predict' espera uma lista de arrays 2D.
    # Vamos assumir que as imagens de entrada são uma stack de imagens e dividi-las
    # para passar para o seu modelo.
    # O seu código de treino sugere que a entrada é uma única imagem por vez.
    # Vamos converter o array 3D (Z, Y, X) em uma lista de arrays 2D (Y, X).
    
    # Por padrão, SimpleITK retorna Z, Y, X. É melhor garantir que seja Y, X, Z.
    # Vamos permutar as dimensões se necessário.
    
    # A sua `predict` espera `list[np.ndarray]`, então vamos converter o array
    # 3D em uma lista de arrays 2D para que a função `predict` possa iterar sobre eles.
    # Assumindo que a dimensão do canal é a última, como em (H, W, C).
    
    # SimpleITK retorna em ordem (Z, Y, X). Sua rede espera (Y, X, C).
    # Precisamos de um array RGB (3 canais) para a normalização.
    # O `load_image_file_as_array` do baseline lê um array 3D e retorna-o. 
    # Sua rede espera imagens 2D ou uma lista delas.
    
    # A maneira mais segura é tratar a entrada como uma única imagem 3D 
    # e converter para o formato que sua rede espera. 
    # O baseline já retorna um único array.
    # Você precisa converter de (Z, Y, X) para (Y, X, C) para que as
    # transformações da torchvision funcionem corretamente.
    
    # Convertendo de (Z, Y, X) para (Z, Y, X, C) para RGB
    # Assumimos que o SimpleITK lê uma imagem com 3 canais (RGB). 
    # Se a imagem for em escala de cinza, ela terá 2 ou 3 dimensões.
    
    if img_array.ndim == 2:
        # Imagem em escala de cinza, converte para 3 canais
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    
    if img_array.ndim == 3 and img_array.shape[-1] != 3:
        # Se for 3D e o último canal não for 3, assumimos que é uma stack
        # de imagens ou Z, Y, X.
        # Vamos tratar como uma única imagem 3D e retornar uma lista de uma imagem
        img_array = np.moveaxis(img_array, 0, -1)  # Tenta mover para o formato HWC
        
    return [img_array]


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
