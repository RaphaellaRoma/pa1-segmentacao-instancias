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
não associa cada imagem a uma modalidade. Essa diferença em relação à regra
literal do enunciado está explicada abaixo.

Testes rápidos que não treinam modelos:

```bash
python -m unittest discover -s tests
```

Os resultados das 18 execuções da Parte 3 estão em `results/part3_*.csv` e
seus gráficos no notebook. As Partes 4–6 foram executadas e estão documentadas
em `task1.ipynb`, com os CSVs e o checkpoint Atrous em `results/` e
`checkpoints/`. Resumo da execução atual:

- Parte 4: a fusão entre tiles elevou o mAP do mosaico de `0,2876` para
  `0,3502`, mas piorou o erro de contagem de `3` para `28` objetos. Portanto,
  a fusão não melhorou as duas métricas.
- Parte 5: a SmallUNet original teve mAP `0,4661` e erro médio de contagem
  `9,89` nas 101 imagens de teste; a versão Atrous teve `0,4307` e `14,80`.
  O campo receptivo maior **não** melhorou este experimento.
- Parte 6: as 101 imagens foram avaliadas nas dez condições (limpa e três
  intensidades de cada corrupção). O mAP limpo foi `0,4661`; na intensidade
  máxima caiu para `0,3109` com blur, `0,3360` com brilho/contraste e
  `0,0018` com ruído.

As flags `RUN_ATROUS_TRAINING` e `RUN_STRESS_TEST` ficam em `False` por padrão
para não repetir um treino ou uma avaliação longa ao abrir o notebook. Mude
para `True` somente se quiser reproduzir esses experimentos. Os números
acima pertencem ao checkpoint atual; resultados históricos de outros treinos
não devem ser misturados na mesma comparação.

Há também uma célula opcional **“Experimento complementar — duração do treino”**
logo após a Parte 2. Ela testa até 20 épocas, partindo de uma nova inicialização
com a mesma configuração da Parte 2, e guarda o melhor modelo até a época 10
e até a 20 **na mesma execução**. Compara os dois apenas na validação.
A flag `RUN_20_EPOCH_EXPERIMENT` está em `True` para a execução na máquina com
GPU. Prepare antes os dados e as funções de treino da Parte 2 no mesmo kernel
e execute **somente a célula desse experimento**; não use “Executar tudo”, que
repetiria os treinos anteriores. Os arquivos separados são
`checkpoints/small_unet_boundary_10ep_control.pt`,
`checkpoints/small_unet_boundary_20ep.pt`, `results/part2_20ep_history.csv`
e `results/part2_20ep_validation.csv`. A célula não roda em `train.py` e não
substitui o checkpoint final; o teste fica reservado até avaliarmos a validação.

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
- `AI_LOG.md`: registro de uso de IA pedido no enunciado, mantido **somente
  local** até a revisão da dupla; ainda não integra o repositório remoto.

Na Parte 5, a época do Atrous foi escolhida pelo conjunto de validação antes
da comparação no teste. A estratificação por modalidade continua sendo uma
limitação: a [fonte oficial BBBC038](https://bbbc.broadinstitute.org/BBBC038)
distribui as imagens por `ImageId` e um metadata agregado por experimento, mas
não fornece aqui uma tabela `ImageId → modalidade`. O split por densidade não
substitui literalmente o pedido do enunciado; para corrigir isso seria preciso
obter ou anotar rótulos confiáveis por imagem e refazer os treinos/avaliações.
