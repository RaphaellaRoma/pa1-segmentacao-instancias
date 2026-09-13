"""Modelo final e decodificação de instâncias do PA1.

Este arquivo é usado por ``inferencia.ipynb`` e pelos testes locais. Ele não
baixa dados nem treina: apenas carrega um checkpoint já produzido na Parte 2.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed


class DoubleConv(nn.Module):
    """Duas convoluções 3x3, cada uma com BatchNorm e ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    """Mesma arquitetura da Parte 2: 1 canal de entrada e 3 classes."""

    def __init__(self, in_channels=1, out_channels=3, base_channels=16):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        x = self.bottleneck(self.pool(skip3))
        x = self.dec3(torch.cat([self.up3(x), skip3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        return self.head(x)


def decode_instances(probabilities, parameters):
    """Fundo/interior/fronteira -> marcadores -> watershed.

    Reproduz exatamente o decoder calibrado na validação na Parte 2.
    """
    foreground = 1.0 - probabilities[0] >= parameters["foreground_threshold"]
    interior = probabilities[1] >= parameters["interior_threshold"]
    interior &= foreground

    markers, _ = label(interior, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(markers.ravel())
    clean_markers = np.zeros(markers.shape, dtype=np.int32)
    next_id = 1
    for marker_id in range(1, len(sizes)):
        if sizes[marker_id] >= parameters["minimum_marker_size"]:
            clean_markers[markers == marker_id] = next_id
            next_id += 1

    if next_id == 1:
        instances, _ = label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
        return instances.astype(np.int32)

    distance = distance_transform_edt(foreground)
    instances = watershed(-distance, markers=clean_markers, mask=foreground)
    return instances.astype(np.int32)


def load_model(checkpoint_path, device=None):
    """Carrega pesos e parâmetros do decoder, sem retreinar."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["decoder_parameters"], device


@torch.inference_mode()
def predict_probabilities(model, image_tensor, device):
    """Calcula as três probabilidades em uma imagem de qualquer tamanho."""
    if image_tensor.ndim != 4 or image_tensor.shape[0:2] != (1, 1):
        raise ValueError("Esperado tensor com formato (1, 1, altura, largura).")
    height, width = image_tensor.shape[-2:]
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    padded = F.pad(image_tensor.to(device), (0, pad_width, 0, pad_height))
    logits = model(padded)[..., :height, :width]
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def prepare_image(image_path):
    """Aplica a mesma conversão para escala de cinza usada no treino."""
    with Image.open(image_path) as image_file:
        pixels = np.asarray(image_file.convert("L"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.copy())[None, None]
    return pixels, tensor


def predict_file(image_path, checkpoint_path):
    """Retorna imagem cinza, máscara inteira e probabilidades para um arquivo."""
    model, parameters, device = load_model(checkpoint_path)
    image, tensor = prepare_image(image_path)
    probabilities = predict_probabilities(model, tensor, device)
    instances = decode_instances(probabilities, parameters)
    return image, instances, probabilities


def colorize_instances(instances):
    """Cor determinística por ID; o fundo permanece preto."""
    from colorsys import hsv_to_rgb

    colored = np.zeros((*instances.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        hue = (int(instance_id) * 0.61803398875) % 1.0
        rgb = hsv_to_rgb(hue, 0.8, 1.0)
        colored[instances == instance_id] = np.array(rgb) * 255
    return colored
