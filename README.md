# Modelo preditivo de rsecisão voluntária

Produto entregue como modelo piloto para teste e coleta de feedback. Trata-se de um modelo preditivo de recisão voluntária.

# Versões

Versão do python utlizada neste projeto: 3.9.12

# Instalação

Utilizar o pacote de gerenciamento [pip](https://pip.pypa.io/en/stable/) para instalar as dependências necessárias. Estas estão presentes em requirements.txt.

```bash
pip install -r /path/to/requirements.txt
```

## Em caso de falha na instalação do requirements.txt

Caso ocorra erro em múltiplas tentativas de instalação das dependência presentes em requirements.txt, deve-se executar a linha de comando abaixo e tentar novamente a instalção do requirements.txt através do pip.

```bash
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"
```

# Diretórios e arquivos

Descrição dos Diretórios e arquivos

## Árvore para pastas e arquivos

```bash
│   abt_prod_data.py
│   abt_training_data.py
│   app.py
│   config.yaml
│   pipeline_preprocess_prod_data.py
│   pipeline_prod_weighted_lgbm.py
│   pipeline_scores_weighted_lgbm.py
│   README.md
│   requirements.txt
│
├───.streamlit
│       config.toml
│
├───database
│       abt_prod_data.csv
│       abt_train_data.csv
│       ativos_2022_08_to_2023_01.xlsx
│       ativos_2022_11_to_2023_01.xlsx
│       ativos_2023_02.xlsx
│       beneficios_2022_08_to_2023_01.xlsx
│       beneficios_2022_11_to_2023_01.xlsx
│       beneficios_2023_02.xlsx
│       dados_gridsearchcv.pkl
│       dados_gridsearchcv_prod.pkl
│       dados_gridsearchcv_train.pkl
│       dados_preprocess_prod.pkl
│       dados_pre_processados.pkl
│       dados_pre_processados_prod.pkl
│       dados_pre_processados_train.pkl
│       dados_raw_prod.pkl
│       dependentes_2022_08_to_2023_01.xlsx
│       dependentes_2022_11_to_2023_01.xlsx
│       dependentes_2023_02.xlsx
│       multiplos_2022_08_to_2023_01.xlsx
│       multiplos_2022_11_to_2023_01.xlsx
│       multiplos_2023_02.xlsx
│
├───encoders_scalers
│       leave_one_out_encoder.pkl
│       leave_one_out_encoder_train.pkl
│       ordinal_encoder.pkl
│       ordinal_encoder_train.pkl
│       standard_scaler_Custo_Benef_Mes_Anterior.pkl
│       standard_scaler_Custo_Benef_Mes_Anterior_train.pkl
│       standard_scaler_Num_Benef_Mes_Anterior.pkl
│       standard_scaler_Num_Benef_Mes_Anterior_train.pkl
│       standard_scaler_num_idade.pkl
│       standard_scaler_num_idade_train.pkl
│       standard_scaler_num_tempo_casa.pkl
│       standard_scaler_num_tempo_casa_train.pkl
│       standard_scaler_qtd_jornada.pkl
│       standard_scaler_qtd_jornada_train.pkl
│       standard_scaler_TotalMesesDesdeUltimoMeritoPromocao.pkl
│       standard_scaler_TotalMesesDesdeUltimoMeritoPromocao_train.pkl
│       standard_scaler_vlr_salario.pkl
│       standard_scaler_vlr_salario_train.pkl
│
├───images
│       cogna_logo.png
│
├───model_metrics
│       lgbmclassifier_accuracy_noopt_scores.pkl
│       lgbmclassifier_accuracy_optfscore_scores.pkl
│       lgbmclassifier_accuracy_optprecisionrecall_scores.pkl
│       lgbmclassifier_accuracy_optroc_scores.pkl
│       lgbmclassifier_f1_noopt_scores.pkl
│       lgbmclassifier_f1_optfscore_scores.pkl
│       lgbmclassifier_f1_optprecisionrecall_scores.pkl
│       lgbmclassifier_f1_optroc_scores.pkl
│       lgbmclassifier_neg_log_loss_noopt_scores.pkl
│       lgbmclassifier_neg_log_loss_optfscore_scores.pkl
│       lgbmclassifier_neg_log_loss_optprecisionrecall_scores.pkl
│       lgbmclassifier_neg_log_loss_optroc_scores.pkl
│       lgbmclassifier_precision_noopt_scores.pkl
│       lgbmclassifier_precision_optfscore_scores.pkl
│       lgbmclassifier_precision_optprecisionrecall_scores.pkl
│       lgbmclassifier_precision_optroc_scores.pkl
│       lgbmclassifier_recall_noopt_scores.pkl
│       lgbmclassifier_recall_optfscore_scores.pkl
│       lgbmclassifier_recall_optprecisionrecall_scores.pkl
│       lgbmclassifier_recall_optroc_scores.pkl
│       lgbmclassifier_roc_auc_noopt_scores.pkl
│       lgbmclassifier_roc_auc_optfscore_scores.pkl
│       lgbmclassifier_roc_auc_optprecisionrecall_scores.pkl
│       lgbmclassifier_roc_auc_optroc_scores.pkl
│       resultados_lgbmclassifier_accuracy_noopt.txt
│       resultados_lgbmclassifier_accuracy_optfscore.txt
│       resultados_lgbmclassifier_accuracy_optprecisionrecall.txt
│       resultados_lgbmclassifier_accuracy_optroc.txt
│       resultados_lgbmclassifier_f1_noopt.txt
│       resultados_lgbmclassifier_f1_optfscore.txt
│       resultados_lgbmclassifier_f1_optprecisionrecall.txt
│       resultados_lgbmclassifier_f1_optroc.txt
│       resultados_lgbmclassifier_neg_log_loss_noopt.txt
│       resultados_lgbmclassifier_neg_log_loss_optfscore.txt
│       resultados_lgbmclassifier_neg_log_loss_optprecisionrecall.txt
│       resultados_lgbmclassifier_neg_log_loss_optroc.txt
│       resultados_lgbmclassifier_precision_noopt.txt
│       resultados_lgbmclassifier_precision_optfscore.txt
│       resultados_lgbmclassifier_precision_optprecisionrecall.txt
│       resultados_lgbmclassifier_precision_optroc.txt
│       resultados_lgbmclassifier_recall_noopt.txt
│       resultados_lgbmclassifier_recall_optfscore.txt
│       resultados_lgbmclassifier_recall_optprecisionrecall.txt
│       resultados_lgbmclassifier_recall_optroc.txt
│       resultados_lgbmclassifier_roc_auc_noopt.txt
│       resultados_lgbmclassifier_roc_auc_optfscore.txt
│       resultados_lgbmclassifier_roc_auc_optprecisionrecall.txt
│       resultados_lgbmclassifier_roc_auc_optroc.txt
│       summary_plot_lgbmclassifier_accuracy_weighted.pdf
│       summary_plot_lgbmclassifier_f1_weighted.pdf
│       summary_plot_lgbmclassifier_neg_log_loss_weighted.pdf
│       summary_plot_lgbmclassifier_precision_weighted.pdf
│       summary_plot_lgbmclassifier_recall_weighted.pdf
│       summary_plot_lgbmclassifier_roc_auc_weighted.pdf
│
└───trained_models
        lgbmclassifier_weighted_accuracy.pkl
        lgbmclassifier_weighted_accuracy_train.pkl
        lgbmclassifier_weighted_f1.pkl
        lgbmclassifier_weighted_f1_train.pkl
        lgbmclassifier_weighted_neg_log_loss.pkl
        lgbmclassifier_weighted_precision.pkl
        lgbmclassifier_weighted_recall.pkl
        lgbmclassifier_weighted_roc_auc.pkl
        rfe_weighted_accuracy.pkl
        rfe_weighted_accuracy_train.pkl
        rfe_weighted_f1.pkl
        rfe_weighted_f1_train.pkl
        rfe_weighted_neg_log_loss.pkl
        rfe_weighted_precision.pkl
        rfe_weighted_recall.pkl
        rfe_weighted_roc_auc.pkl
```

## Diretórios

    .streamlit: contém config.toml, onde estão as configurações de layout e cores do app.

    database: contém todos os dados envolvido em pré-processamento e processamento.

    encoders_scalres: contém todos os arquivos de encoders e scalers para os cenários de treino e produção.

    images: contém imagens utilizadas no aplicativo.

    model_metrics: contém os arquivos de avaliação do modelo com as principais métricas utilizadas no app.

    trained_models: contém os modelos treinados para utilização em produção.

## Arquivos

    abt_training_data.py: script para ingestão dos dados e aplicação das regras de negócio, tendo como output uma tabela analítica com os campos selecionados para pré-processamento.

    abt_prod_data.py: script com as mesas características de "abt_training_data.py", porém realiza a ingestão dos dados que serão base para as predições do modelo. Trata-se da base de produção que será visualizada pelos gestores.

    pipeline_prod_weighted_lgbm.py: script responsável pelo treinamento do modelo utilizado em produção. O pipeline presente neste script já foi selecionado como melhor candidato para treinamento do modelo.

    pipeline_preprocess_prod_data.py: script que contém o pipeline de pré-processamento dos dados gerados por "abt_prod_data.py". Este script só deve ser executado após a execução de "pipeline_prod_weighted_lgbm.py", uma vez que os encoders necessários para pré-processamento da base de produção são treinados neste último.

    pipeline_scores_weighted_lgbm.py: pipeline de treinamento e avaliação para geração de pipelines candidatos a produção. Nele encontram-se todos os cenários implementados no projeto para avaliação do melhor pipeline de pré-processamento, processamento e otimização do modelo final.

    app.py: contém o código fonte para a aplicação. Foi utilizado o framework streamlit.

    config.yaml: contém as credenciais de acesso para cada usuário selecionado para participar do piloto.

# Modo de uso

## Sequência de execução para treinamento do modelo e execução do app

    1. abt_training_data
    2. abt_prod_data.py
        2.1. pipeline_score_weighted_lgbm.py: escolher o melhor pipeline dentre todos possíveis
    3. pipeline_prod_weighted_lgbm.py: com o melhor pipeline gerado por 2.1
    4. pipeline_preprocess_prod_data.py
    5. app.py

Para executar os scripts acima podemos utilizar

```bash
python [nome_do_script.py]
```

Exemplo:

```bash
python pipeline_score_weighted_lgbm.py
```

Nota: nenhum dos scripts aceita parâmetros passados pelo usuário. 

## Arquitetura do produto

![plot](./images/pipeline_modelo_preditivo_turnover.drawio.png)

## Implementação em produção

Para colocar o app em produção e disponibilizar portas de acesso basta executar o comando abaixo no terminal dentro do diretório do projeto.

```bash
streamlit run app.py
```
Ao fazer isso duas URL's serão geradas. Para desenvolvimento e apresentações pode-se utilizar "Local URL", onde apenas o usuário que está com o aplicativo sendo executado,
em sua própria máquina terá acesso. Para que usuários da MESMA rede acessem o aplicativo basta disponibilizar o link dado por "Network URL".

## Dataset utilizado

Foi utilizado um dataset público e anonimizado.

Descrição dos dados:

- "departamento" - o departamento a que o empregado pertence.
- "promovido" - 1 se o empregado foi promovido nos 24 meses anteriores, 0 caso contrário.
- "revisão" - a pontuação composta que o funcionário recebeu na sua última avaliação.
- "projectos" - quantos projectos o funcionário está envolvido.
- "salário" - por razões de confidencialidade, o salário vem em três níveis: baixo, médio, alto.
- "permanência" - quantos anos o funcionário esteve na empresa.
- "satisfação" - uma medida de satisfação do empregado, a partir de questionários.
- "bonus" - 1 se o empregado recebeu um bonus nos 24 meses anteriores, 0 caso contrário.
- "avg_hrs_month" - a média de horas que o empregado trabalhou num mês.
- "left" - "sim" se o empregado acabou por sair, "não" caso contrário.

Fonte das informações e dados: https://www.kaggle.com/datasets/marikastewart/employee-turnover

# Suporte e feedback

Qualquer dúvida ou sugestão entrar em contato com peopleanalytics@cogna.com.br.