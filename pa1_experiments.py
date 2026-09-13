"""Funções de avaliação das Partes 4, 5 e 6 do PA1.

O split e a regra de mAP são os mesmos de ``task1.ipynb``. Mantemos funções
pequenas para que cada etapa possa ser testada separadamente.
"""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import find_objects
from sklearn.model_selection import train_test_split

from pa1_inference import decode_instances, predict_probabilities


IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def find_data_root(data_root):
    """Localiza as 670 amostras; se preciso, extrai o ZIP local uma só vez."""
    data_root = Path(data_root)
    candidates = (data_root / "stage1_train", data_root)

    def count_samples(root):
        if not root.is_dir():
            return 0
        return sum(
            1 for path in root.iterdir() if path.is_dir()
            and (path / "images").is_dir() and (path / "masks").is_dir()
        )

    for candidate in candidates:
        if count_samples(candidate) == 670:
            return candidate

    archive_path = data_root / "stage1_train.zip"
    if archive_path.is_file():
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("O ZIP do dataset está corrompido.")
            first_parts = {
                Path(name).parts[0] for name in archive.namelist()
                if Path(name).parts
            }
            has_wrapper = first_parts == {"stage1_train"}
            extraction_root = data_root if has_wrapper else data_root / "stage1_train"
            extraction_root.mkdir(parents=True, exist_ok=True)
            archive.extractall(extraction_root)
        for candidate in candidates:
            if count_samples(candidate) == 670:
                return candidate
    raise FileNotFoundError(
        f"Não encontrei 670 amostras válidas em {data_root}. "
        "Verifique stage1_train.zip ou execute a célula de dados da Parte 1."
    )


def build_catalog(data_root):
    """Recria o catálogo usado no split da Parte 1, sem carregar máscaras."""
    root = find_data_root(data_root)
    records = []
    for sample_dir in sorted(root.iterdir()):
        if not (sample_dir / "images").is_dir():
            continue
        image_path = next((sample_dir / "images").glob("*.png"))
        with Image.open(image_path) as image:
            width, height = image.size
        count = len(list((sample_dir / "masks").glob("*.png")))
        records.append({
            "image_id": sample_dir.name,
            "width": width,
            "height": height,
            "num_instances": count,
            "instances_per_100k_pixels": count / (width * height) * 100_000,
        })
    catalog = pd.DataFrame(records)
    assert len(catalog) == 670 and catalog["image_id"].is_unique
    return catalog


def split_catalog(catalog, seed=42):
    """Mesmo split estratificado 70/15/15 por densidade do notebook."""
    data = catalog.copy()
    names = ["muito baixa", "baixa", "média", "alta", "muito alta"]
    data["density_bin"] = pd.qcut(
        data["instances_per_100k_pixels"].rank(method="first"),
        q=5, labels=names,
    )
    train, remaining = train_test_split(
        data, test_size=0.30, random_state=seed, stratify=data["density_bin"]
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=seed,
        stratify=remaining["density_bin"],
    )
    return tuple(
        part.sort_values("image_id").reset_index(drop=True)
        for part in (train, validation, test)
    )


def load_sample(data_root, image_id):
    """Lê uma imagem e junta seus PNGs individuais em IDs 1, 2, ..."""
    sample_dir = Path(data_root) / image_id
    image_path = next((sample_dir / "images").glob("*.png"))
    with Image.open(image_path) as image_file:
        image = np.asarray(image_file.convert("L"), dtype=np.float32) / 255.0
    instances = np.zeros(image.shape, dtype=np.int32)
    for instance_id, mask_path in enumerate(
        sorted((sample_dir / "masks").glob("*.png")), start=1
    ):
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(mask_file.convert("L")) > 0
        if mask.shape != image.shape or np.any(mask & (instances > 0)):
            raise ValueError(f"Máscara inválida ou sobreposta em {image_id}")
        instances[mask] = instance_id
    return image, instances


def instance_iou_matrix(true_instances, predicted_instances):
    """IoU de cada par (real, previsto), sem usar mAP pronta de biblioteca."""
    true_count = int(true_instances.max())
    pred_count = int(predicted_instances.max())
    if true_count == 0 or pred_count == 0:
        return np.zeros((true_count, pred_count), dtype=np.float32)
    true_flat = true_instances.ravel().astype(np.int64)
    pred_flat = predicted_instances.ravel().astype(np.int64)
    pair_index = true_flat * (pred_count + 1) + pred_flat
    intersections = np.bincount(
        pair_index, minlength=(true_count + 1) * (pred_count + 1)
    ).reshape(true_count + 1, pred_count + 1)[1:, 1:]
    true_areas = np.bincount(true_flat, minlength=true_count + 1)[1:]
    pred_areas = np.bincount(pred_flat, minlength=pred_count + 1)[1:]
    unions = true_areas[:, None] + pred_areas[None, :] - intersections
    return intersections / np.maximum(unions, 1)


def calculate_instance_metrics(true_instances, predicted_instances):
    """Matching guloso por IoU; AP=TP/(TP+FP+FN), média em 0,50:0,95."""
    true_count = int(true_instances.max())
    pred_count = int(predicted_instances.max())
    iou = instance_iou_matrix(true_instances, predicted_instances)
    scores = []
    for threshold in IOU_THRESHOLDS:
        true_indices, pred_indices = np.where(iou >= threshold)
        order = np.argsort(iou[true_indices, pred_indices])[::-1]
        matched_true = set()
        matched_pred = set()
        for pair in order:
            true_id = int(true_indices[pair])
            pred_id = int(pred_indices[pair])
            if true_id not in matched_true and pred_id not in matched_pred:
                matched_true.add(true_id)
                matched_pred.add(pred_id)
        tp = len(matched_true)
        denominator = true_count + pred_count - tp
        scores.append(tp / denominator if denominator else 1.0)
    return {
        "map": float(np.mean(scores)),
        "true_count": true_count,
        "predicted_count": pred_count,
        "count_error": abs(true_count - pred_count),
    }


def predict_sample(model, parameters, device, image):
    """Uma imagem 2D já normalizada -> instâncias e probabilidades."""
    tensor = torch.from_numpy(image.copy())[None, None]
    probabilities = predict_probabilities(model, tensor, device)
    return decode_instances(probabilities, parameters), probabilities


def evaluate_ids(data_root, image_ids, model, parameters, device):
    """Avalia cada imagem separadamente; não altera nenhum parâmetro."""
    records = []
    for image_id in image_ids:
        image, truth = load_sample(data_root, image_id)
        prediction, _ = predict_sample(model, parameters, device, image)
        metrics = calculate_instance_metrics(truth, prediction)
        records.append({"image_id": image_id, **metrics})
    return pd.DataFrame(records)


def object_diameters(data_root, image_ids):
    """Maior lado da caixa delimitadora de cada núcleo (em pixels)."""
    diameters = []
    for image_id in image_ids:
        _, instances = load_sample(data_root, image_id)
        for region in find_objects(instances):
            if region is not None:
                height = region[0].stop - region[0].start
                width = region[1].stop - region[1].start
                diameters.append(max(height, width))
    return np.asarray(diameters, dtype=np.int32)


def make_mosaic(data_root, image_ids, square_size=256):
    """Junta quatro imagens diferentes em 2x2, sem repetir IDs da verdade.

    Recortamos o centro de cada imagem para que o mosaico tenha tamanho fixo.
    Isso facilita observar núcleos cortados pelos limites dos *tiles* de
    inferência, que não coincidem necessariamente com as quatro emendas.
    """
    if len(image_ids) != 4:
        raise ValueError("O mosaico 2x2 precisa de exatamente quatro imagens.")
    image_mosaic = np.zeros((square_size * 2, square_size * 2), dtype=np.float32)
    truth_mosaic = np.zeros(image_mosaic.shape, dtype=np.int32)
    next_id = 0
    for index, image_id in enumerate(image_ids):
        image, truth = load_sample(data_root, image_id)
        height, width = image.shape
        pad_h = max(0, square_size - height)
        pad_w = max(0, square_size - width)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)))
            truth = np.pad(truth, ((0, pad_h), (0, pad_w)))
        top = (image.shape[0] - square_size) // 2
        left = (image.shape[1] - square_size) // 2
        image = image[top:top + square_size, left:left + square_size]
        truth = truth[top:top + square_size, left:left + square_size].copy()
        # Alguns núcleos podem sair inteiros no recorte. Renumeramos os restantes.
        truth = relabel_consecutive(truth)
        truth[truth > 0] += next_id
        next_id = int(truth.max())
        row, column = divmod(index, 2)
        rows = slice(row * square_size, (row + 1) * square_size)
        columns = slice(column * square_size, (column + 1) * square_size)
        image_mosaic[rows, columns] = image
        truth_mosaic[rows, columns] = truth
    return image_mosaic, truth_mosaic


def relabel_consecutive(instances):
    """Transforma IDs possivelmente esparsos em 1, 2, 3, ...; mantém fundo 0."""
    result = np.zeros(instances.shape, dtype=np.int32)
    positive_ids = np.unique(instances)
    positive_ids = positive_ids[positive_ids > 0]
    for new_id, old_id in enumerate(positive_ids, start=1):
        result[instances == old_id] = new_id
    return result


def tile_starts(length, tile_size, stride):
    """Inclui sempre o último tile: a borda da imagem não fica sem cobertura."""
    if not 0 < stride <= tile_size or tile_size > length:
        raise ValueError("Use 0 < stride <= tile_size <= tamanho da imagem.")
    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def predict_tiles(model, parameters, device, image, tile_size=256, stride=192):
    """Uma única predição por tile; ambas as fusões usam exatamente estes mapas."""
    height, width = image.shape
    tiles = []
    probability_sum = np.zeros((3, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.int32)
    for top in tile_starts(height, tile_size, stride):
        for left in tile_starts(width, tile_size, stride):
            crop = image[top:top + tile_size, left:left + tile_size]
            tensor = torch.from_numpy(crop.copy())[None, None]
            probabilities = predict_probabilities(model, tensor, device)
            local_instances = decode_instances(probabilities, parameters)
            tiles.append({"top": top, "left": left, "instances": local_instances})
            rows = slice(top, top + tile_size)
            columns = slice(left, left + tile_size)
            probability_sum[:, rows, columns] += probabilities
            coverage[rows, columns] += 1
    assert np.all(coverage > 0), "Há pixels do mosaico sem inferência."
    return tiles, probability_sum / coverage[None], coverage


def naive_stitch(tiles, image_shape):
    """Cola IDs locais sem reconciliar objetos repetidos na sobreposição."""
    canvas = np.zeros(image_shape, dtype=np.int32)
    next_id = 0
    for tile in tiles:
        local = tile["instances"].copy()
        local[local > 0] += next_id
        top, left = tile["top"], tile["left"]
        height, width = local.shape
        crop = canvas[top:top + height, left:left + width]
        crop[local > 0] = local[local > 0]
        next_id += int(tile["instances"].max())
    return relabel_consecutive(canvas)


class UnionFind:
    """Registra que dois IDs de tiles diferentes representam o mesmo núcleo."""

    def __init__(self):
        self.parent = {}

    def find(self, object_id):
        self.parent.setdefault(object_id, object_id)
        if self.parent[object_id] != object_id:
            self.parent[object_id] = self.find(self.parent[object_id])
        return self.parent[object_id]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def merge_stitch(tiles, image_shape, minimum_overlap_iou=0.15):
    """Une IDs que cobrem o mesmo núcleo na faixa de sobreposição.

    A IoU é calculada **somente nos pixels vistos pelos dois tiles**. Se
    usássemos a área inteira de um objeto local, um núcleo cortado pela borda
    teria uma IoU artificialmente baixa e a fusão falharia.
    """
    canvas = np.zeros(image_shape, dtype=np.int32)
    covered = np.zeros(image_shape, dtype=bool)
    groups = UnionFind()
    next_id = 0
    for tile in tiles:
        local = tile["instances"].copy()
        local[local > 0] += next_id
        top, left = tile["top"], tile["left"]
        height, width = local.shape
        rows = slice(top, top + height)
        columns = slice(left, left + width)
        old = canvas[rows, columns]
        overlap = covered[rows, columns]

        old_ids = np.unique(old[overlap & (old > 0)])
        new_ids = np.unique(local[overlap & (local > 0)])
        for old_id in old_ids:
            old_mask = (old == old_id) & overlap
            for new_id in new_ids:
                new_mask = (local == new_id) & overlap
                intersection = np.count_nonzero(old_mask & new_mask)
                if intersection == 0:
                    continue
                union = np.count_nonzero(old_mask | new_mask)
                if intersection / union >= minimum_overlap_iou:
                    groups.union(int(old_id), int(new_id))

        empty_pixels = (old == 0) & (local > 0)
        old[empty_pixels] = local[empty_pixels]
        covered[rows, columns] = True
        next_id += int(tile["instances"].max())

    # A união altera os IDs, não os pixels. No fim aplicamos a equivalência.
    merged = np.zeros(image_shape, dtype=np.int32)
    for old_id in np.unique(canvas):
        if old_id == 0:
            continue
        merged[canvas == old_id] = groups.find(int(old_id))
    return relabel_consecutive(merged)


def encoder_receptive_field(bottleneck_dilation=1):
    """Campo receptivo teórico no fim do encoder da SmallUNet.

    ``jump`` é a distância, em pixels da entrada, entre posições vizinhas do
    mapa atual. As três camadas MaxPool dobram esse valor; as duas convoluções
    do gargalo podem ter dilatação 1 (original) ou 2 (Atrous).
    """
    receptive_field = 1
    jump = 1
    for block in range(4):
        dilation = bottleneck_dilation if block == 3 else 1
        for _ in range(2):
            receptive_field += (3 - 1) * dilation * jump
        if block < 3:
            receptive_field += (2 - 1) * jump
            jump *= 2
    return receptive_field, jump


def corrupt_image(image, kind, severity, seed=42):
    """Corrupções fixas e reproduzíveis; a máscara verdadeira não muda."""
    if kind == "limpa":
        return image.copy()
    if severity not in (1, 2, 3):
        raise ValueError("A severidade deve ser 1, 2 ou 3.")
    if kind == "ruido":
        rng = np.random.default_rng(seed)
        standard_deviation = (0.05, 0.10, 0.15)[severity - 1]
        return np.clip(
            image + rng.normal(0, standard_deviation, image.shape), 0, 1
        ).astype(np.float32)
    if kind == "blur":
        radius = (0.7, 1.4, 2.1)[severity - 1]
        image_pil = Image.fromarray((image * 255).astype(np.uint8))
        blurred = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(blurred, dtype=np.float32) / 255.0
    if kind == "brilho_contraste":
        # Escurece e comprime o contraste em torno da média da imagem.
        factor = (0.8, 0.6, 0.4)[severity - 1]
        mean = float(image.mean())
        return np.clip((image - mean) * factor + mean * factor, 0, 1)
    raise ValueError(f"Corrupção desconhecida: {kind}")
