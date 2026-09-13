# PA1 — Segmentação de instâncias

Projeto da disciplina de Aprendizado Profundo. Usamos uma Small U-Net para
segmentar núcleos do BBBC038v1. A baseline binária vira instâncias por
componentes conexos; o modelo final prevê fundo/interior/fronteira e usa
watershed. A regra de matching e o mAP foram implementados por nós.

## Ambiente e dados

Use Python 3.12 ou um ambiente Colab com PyTorch. Na raiz do repositório:

```bash
python -m pip install -r requirements.txt
```

O repositório contém `data/BBBC038/stage1_train.zip` e `metadata.xlsx`.
Ao executar a célula de dados de `task1.ipynb`, o código usa as 670 amostras
já extraídas, ou extrai o ZIP local; só tenta Drive/Broad se o ZIP não existir.
Também é possível obter os dados da [fonte oficial BBBC038](https://bbbc.broadinstitute.org/BBBC038).
As imagens PNG extraídas não são versionadas, mas o ZIP é.

## Um comando para treinar; um para avaliar

Execute na raiz do projeto, com um kernel `python3` registrado no ambiente:

```bash
python train.py
python evaluate.py
```

`train.py` executa as Partes 0–2 do notebook, salva
`task1_treinado.ipynb` e atualiza o checkpoint final em `checkpoints/`.
Esse comando pode demorar; recomenda-se GPU. `evaluate.py` **não treina**:
reconstrói o split estratificado com seed 42, carrega o checkpoint e grava o
mAP e o erro de contagem por imagem em `results/final_test_per_image.csv`.
Se o ZIP estiver presente mas as PNGs não, ele extrai o ZIP local uma vez.
O split histórico é por **densidade**, pois o metadata fornecido é agregado e
não associa cada imagem a uma modalidade; essa diferença em relação à regra
literal do enunciado está explicada em `STUDY_NOTES.md`.

Testes rápidos que não treinam modelos:

```bash
python -m unittest discover -s tests
```

Os resultados das 18 execuções da Parte 3 estão em `results/part3_*.csv` e
seus gráficos no notebook. As Partes 4–6 também estão em `task1.ipynb`.
A Parte 5 tem um treino adicional Atrous e a Parte 6 tem dez condições de
teste; por segurança, mude `RUN_ATROUS_TRAINING` e `RUN_STRESS_TEST` para
`True` nas respectivas células **quando estiver na máquina que vai executá-los**.
Essas avaliações ainda precisam ser rodadas para produzir tabelas finais.

## Inferência em qualquer imagem

Abra `inferencia.ipynb`, ajuste somente `IMAGE_PATH` e execute as células.
Ele carrega `checkpoints/small_unet_boundary_watershed_v2.pt`, mostra a
probabilidade de fronteira e devolve a máscara colorida e a contagem sem
retreinar. A máscara é salva em `results/<nome>_instancias.png`.

## Arquivos principais

- `task1.ipynb`: explicações, treinos e experimentos das Partes 0–6.
- `task1_2.ipynb`: cópia **local** da contribuição original da colega,
  preservada fora do Git; não é necessária para reproduzir a versão integrada.
- `pa1_inference.py`: arquitetura final, checkpoint e decoder.
- `pa1_experiments.py`: split, métricas, mosaico, campo receptivo e corrupções.
- `STUDY_NOTES.md`: notas de estudo e histórico das decisões (mantidas locais).
- `AI_LOG.md`: registro de uso de IA pedido no enunciado.

Os resultados numéricos das Partes 4–6 não são afirmados como concluídos até
que as respectivas células sejam executadas. O teste da Parte 5 é avaliado
apenas depois de escolher a época pelo conjunto de validação.
