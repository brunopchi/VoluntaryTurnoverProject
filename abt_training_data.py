# Basic libs
import warnings
import pandas as pd
import numpy as np

# Visualization
import matplotlib
import matplotlib.pyplot as plt

# ------------------------------- Raw data ------------------------------- 

# Data from 2022 to 2023
df_ativos = pd.read_excel('./database/ativos_2022_08_to_2023_01.xlsx', dtype={'cod_chapa_funcionario' : object})
df_beneficios = pd.read_excel('./database/beneficios_2022_08_to_2023_01.xlsx', dtype={'cod_chapa_funcionario' : object})
df_dependentes = pd.read_excel('./database/dependentes_2022_08_to_2023_01.xlsx', dtype={'cod_chapa_funcionario' : object})
df_multiplos = pd.read_excel('./database/multiplos_2022_08_to_2023_01.xlsx', dtype={'cod_chapa_funcionario' : object})

# ------------------------- Minimal data quality -------------------------

# Data from 2022 to 2023
df_ativos = df_ativos[df_ativos['cod_chapa_funcionario'].notna()]
df_beneficios = df_beneficios[df_beneficios['cod_chapa_funcionario'].notna()]
df_dependentes = df_dependentes[df_dependentes['cod_chapa_funcionario'].notna()]
df_multiplos = df_multiplos[df_multiplos['cod_chapa_funcionario'].notna()]

# --------------------------- Build fact table ---------------------------

# Data from 2022 to 2023
df = df_ativos.merge(df_beneficios, how='left', on=['Ano/Mês', 'cod_chapa_funcionario'])
df = df.merge(df_dependentes, how='left', on=['Ano/Mês', 'cod_chapa_funcionario'])
df = df.merge(df_multiplos, how='left', on=['Ano/Mês', 'cod_chapa_funcionario'])

# -------------------- Consodlidation of the dataset ---------------------

# Sort the dataframe
df = df.sort_values(['ind_demitido_mes', 'cod_chapa_funcionario', 'Ano/Mês'],
                    ascending=[True, True, True])\
                    .reset_index(drop=True)

# ----------------- Filter to remove duplicated entries ------------------

# Filters cod_chapa leaving only the last record (most recent due to sorting)
df = df.drop_duplicates(subset=['cod_chapa_funcionario'], keep='last').reset_index(drop=True)

# --------------------------- Business rules -----------------------------

# Filtering only administrative employees who were not fired by the employer
selection = (df['tip_funcionario'] == 'ADMINISTRATIVO') & \
            (df['dsc_iniciativa_desligamento'] != 'Empregador') & \
            (df['tip_demissao'] != 'Transferência sem ônus p/ Cedente')
df = df[selection]

# Excludes employees dismissed outside the time period recorded in the dataset 
# (employees who requested termination before month 1 of 2022)
selection = (df['tip_demissao'] != 'Inic.Empregado sem justa causa') | \
            (df['ind_demitido_mes'] != 0.0)
df = df[selection]

# Excludes employees who are active but do not have a record of the most recent 
# month present in the dataset (month 5) Note: Change the Year/Month to the most 
# recent for the period chosen in the data slice
selection = (df['Ano/Mês'] == '2023/01') | (df['dsc_iniciativa_desligamento'] != 'NAO INFORMADO')
df = df[selection]

# ------------------ Variable num_tempo_casa (months) --------------------

df['Ano/Mês'] = df['Ano/Mês'].astype('datetime64[ns]')
df['num_tempo_casa'] = ((df['Ano/Mês'] - df['dat_admissao'])/np.timedelta64(1, 'M'))
df['num_tempo_casa'] = df['num_tempo_casa'].astype(int)

# ---------------------- To prevent data Leakage -------------------------

# Drop columns
df.drop(
    columns=[
        'Ano/Mês',
        'tip_demissao',
        'nom_motivo_demissao',
        'dat_demissao',
        'dat_desligamento',
        'ind_demitido_mes',
        'cod_chapa_funcionario',
        'num_tempo_casa_faixas',
        'tip_funcionario',
        'Total FUNC FAT DEPEN',
        'dat_admissao',
        'dsc_estrutura_n1',
        'dsc_estrutura_n2',
        'Multiplo'
        ],
    axis=1, 
    inplace=True
)

# Drop possible duplicated data
df.drop_duplicates(keep='last', inplace=True)
df.reset_index(drop=True, inplace=True)

# Defining columns order
df = df[[
    'nom_funcao',
    'nom_grupo_cargos',
    #'dsc_estrutura_n1',
    #'dsc_estrutura_n2',
    'nom_empresa',
    'nom_vp',
    'nom_dir_sr',
    'nom_dir',
    'nom_ger_sr',
    'nom_ger',
    'num_idade',
    'num_tempo_casa',
    'nom_sexo',
    'num_nivel_hierarquico',
    'nom_posicao',
    'nom_raca_cor',
    'dsc_modelo_trabalho',
    'nom_estado_civil',
    'nom_grau_instrucao',
    'vlr_salario',
    'qtd_jornada',
    'TotalMesesDesdeUltimoMeritoPromocao',
    'Custo_Benef_Mes_Anterior',
    'Num_Benef_Mes_Anterior',
    'dsc_iniciativa_desligamento'
]]

# Saves data in database
file_name = './database/abt_train_data.csv'
df.to_csv(file_name, index=False)

print(f'Analytic base table saved in {file_name}!')
print(list(df.columns))
print(df.shape)