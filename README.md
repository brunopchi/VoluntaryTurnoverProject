# Modelo preditivo de rescisão voluntária

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
INCLUIR!
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
- "projetos" - quantos projetos o funcionário está envolvido.
- "salário" - por razões de confidencialidade, o salário vem em três níveis: baixo, médio, alto.
- "permanência" - quantos anos o funcionário esteve na empresa.
- "satisfação" - uma medida de satisfação do empregado, a partir de questionários.
- "bonus" - 1 se o empregado recebeu um bonus nos 24 meses anteriores, 0 caso contrário.
- "avg_hrs_month" - a média de horas que o empregado trabalhou num mês.
- "left" - "sim" se o empregado acabou por sair, "não" caso contrário.

Fonte das informações e dados: https://www.kaggle.com/datasets/marikastewart/employee-turnover

# Suporte e feedback

Qualquer dúvida ou sugestão entrar em contato com brunopchimetta@gmail.com.