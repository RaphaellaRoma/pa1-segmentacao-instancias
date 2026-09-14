# Relatório de uso de IA — PA1

## Objetivo e forma de uso

Usamos o Codex como ferramenta de apoio à programação, à integração do código e ao estudo dos conceitos envolvidos no PA1. Recorremos à ferramenta para discutir alternativas de implementação, esclarecer dúvidas e revisar interpretações dos resultados. As decisões sobre o experimento, a execução dos modelos e a análise do que seria apresentado permaneceram sob nossa responsabilidade. Não usamos modelos prontos de segmentação de instâncias nem uma implementação pronta da métrica de AP de instâncias.

## Desenvolvimento da implementação

Na Parte 0, usamos o assistente para estruturar o gerador de imagens sintéticas, o dataset e a Small U-Net binária. Ao estudar o código, pedimos explicações detalhadas sobre `np.ogrid`, índices de matriz, máscaras binárias e de instância, a arquitetura da U-Net e a função dos testes com `assert`. Essas dúvidas orientaram ajustes nas explicações e nas notas de estudo.

Na Parte 1, implementamos a baseline binária para o BBBC038 e a conversão da máscara prevista em objetos por componentes conexos. Com apoio do Codex, escrevemos o cálculo de IoU entre pares de instâncias e o matching guloso por IoU decrescente, em vez de usar uma métrica pronta. Comparamos as métricas semânticas com o mAP e o erro de contagem para entender por que uma boa máscara binária ainda pode unir núcleos diferentes.

Na Parte 2, seguimos a trilha de fronteiras e watershed. O assistente ajudou a organizar os alvos de três classes, o treino e o decoder. A primeira combinação de pesos de classe e pós-processamento produziu supersegmentação; examinamos esse resultado, moderamos os pesos e calibramos os limiares e o tamanho mínimo dos marcadores **somente na validação**. Registramos a tentativa inicial e a correção, em vez de apresentar apenas a versão final.

Na Parte 3, usamos o Codex para organizar as ablações de loss focal, balanceamento das classes e contexto global. Executamos as configurações com duas sementes e mantivemos os resultados por execução em CSV. A comparação foi baseada nas métricas de validação, incluindo média e desvio entre as sementes.

Também reunimos no notebook principal uma parte do trabalho que havia sido desenvolvida em `task1_2.ipynb`: inferência em mosaico, fusão de instâncias entre tiles, análise do campo receptivo, galeria de falhas, variante Atrous e teste de estresse. O Codex auxiliou na integração, na correção da cobertura das bordas dos tiles, na adaptação dos testes ao modelo de três classes e na revisão da variante Atrous para manter o mesmo *output stride*. Criamos ainda `inferencia.ipynb` e scripts para treinar e avaliar sem depender da execução manual de todo o notebook.

## Conferência dos resultados e decisões

Executamos as Partes 4–6 em outra máquina e conferimos as saídas do notebook e os CSVs. A fusão entre tiles elevou o mAP do mosaico, mas piorou seu erro de contagem; a variante Atrous aumentou o campo receptivo teórico, porém não superou a Small U-Net original; e o ruído mais intenso causou grande queda de mAP. Discutimos esses resultados com o assistente e os registramos como limitações, sem descrevê-los como melhorias gerais.

Depois questionamos se aumentar o número de épocas poderia melhorar o modelo da Parte 2. Com ajuda do Codex, preparamos uma comparação controlada: no mesmo treino, guardamos o melhor checkpoint até a época 10 e o melhor até a época 20. Executamos o experimento em outra máquina e versionamos os dois checkpoints e os CSVs. Na validação, o mAP passou de `0,4522` para `0,4825`, e o erro médio de contagem de `11,37` para `10,99`; o melhor checkpoint do treino longo foi selecionado na época 15. **Não interpretamos esse ganho de validação como ganho no teste** e não substituímos o checkpoint final com base apenas nele.

O assistente também ajudou a organizar um guia de implementação e um dicionário das funções do projeto para apoiar nosso estudo do código. Essa documentação foi motivada por dúvidas reais sobre o funcionamento dos dados, da rede, das métricas e da inferência.
