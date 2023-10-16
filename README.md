# Modelo preditivo de rescisão voluntária

Aplicativo criado para o projeto de TCC da MBA USP/ESALQ em Data Science & Analytics. Trata-se de um modelo preditivo de recisão voluntária.

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

## Estrutura das pastas do repositório

```bash
├── database
├── encoders_scalers
├── exploratory_analysis
│   └── data_profiling
│       └── kaggle_voluntary_turnover_1_plots
│           ├── AutoViz
│           └── left
├── model_metrics
└── trained_models
```

## Diretórios

    .streamlit: contém config.toml, onde estão as configurações de layout e cores do app.

    database: contém todos os dados envolvido em pré-processamento e processamento.

    encoders_scalres: contém todos os arquivos de encoders e scalers para os cenários de treino e produção.

    model_metrics: contém os arquivos de avaliação do modelo com as principais métricas utilizadas no app.

    trained_models: contém os modelos treinados para utilização em produção.

## Arquivos (diretório raiz)

    pipeline_scores_weighted_lgbm.py: pipeline de treinamento e avaliação para geração de modelos candidatos a produção. Nele encontram-se todos os cenários implementados no projeto para avaliação do melhor pipeline de pré-processamento, processamento e otimização do modelo final.

    app.py: contém o código fonte para a aplicação. Foi utilizado o framework streamlit.

    config.yaml: contém as credenciais de acesso para cada usuário selecionado para participar do piloto.

    requirements.txx: todas as bibliotecas necessárias para a execução do projeto.

# Modo de uso

## Sequência de execução para treinamento do modelo e execução do app

    1. pipeline_score_weighted_lgbm.py: executa todas as combinações programadas de métricas e modelos.
    2. Escolher o melhor pipeline gerado por 1. e importar em app.py.
    3. Executar app.py.

Para executar os scripts acima podemos utilizar

```bash
python [nome_do_script.py]
```

Exemplo:

```bash
python pipeline_score_weighted_lgbm.py
```

Nota: nenhum dos scripts aceita parâmetros passados pelo usuário. 

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