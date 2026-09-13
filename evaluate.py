"""Avalia o checkpoint final no split de teste sem retreinar.

Uso: python evaluate.py
"""

from pathlib import Path

from pa1_experiments import build_catalog, evaluate_ids, find_data_root, split_catalog
from pa1_inference import load_model


data_root = find_data_root(Path("data/BBBC038"))
catalog = build_catalog(data_root)
_, _, test_catalog = split_catalog(catalog, seed=42)
model, decoder, device = load_model(
    "checkpoints/small_unet_boundary_watershed_v2.pt"
)

results = evaluate_ids(
    data_root, test_catalog["image_id"].tolist(), model, decoder, device
)
output = Path("results/final_test_per_image.csv")
output.parent.mkdir(exist_ok=True)
results.to_csv(output, index=False)

print("Dispositivo:", device)
print("Imagens de teste:", len(results))
print(f"mAP de instâncias: {results['map'].mean():.4f}")
print(f"Erro médio absoluto de contagem: {results['count_error'].mean():.4f}")
print("Resultados por imagem:", output.resolve())
