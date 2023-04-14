# Basic libs
import pickle
import yaml
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Interactive
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from st_aggrid import (
    GridOptionsBuilder, 
    AgGrid, 
    GridUpdateMode, 
    DataReturnMode
)

# Authentication
import streamlit_authenticator as stauth

# Downloading option
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

# Visualizations
import shap
from PIL import Image

# --------------------------------- Functions ---------------------------------

def apply_ordinal_encoder(ordinal_encoder, df_x_test, ordinal_columns):
    """
    Apply ordinal encode from scikit-learn for a given list of columns and categories. 
    For more information check the link below:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    Numpy:
        import numpy as np
    Scikit-Learn:
        from sklearn.preprocessing import OrdinalEncoder
    
    Parameters
    ----------
    ordinal_encoder: sklearn.preprocessing._encoders.OrdinalEncoder
        Trained encoder. Saved in './transformadores_scalers/'
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
    ordinal_columns : list
        List containing the names of all columns to be transformed. Preferably 
        composed of ordinal variables. 
    
    Returns
    -------
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (testing set).  
    """
    
    # Apply the transformation in train and test data
    encoded_array_test = ordinal_encoder.transform(df_x_test.loc[:,ordinal_columns]) 

    # Build the dataframes preserving original indexes
    df_x_test[ordinal_columns] = pd.DataFrame(encoded_array_test,columns=ordinal_encoder.get_feature_names_out(), index=df_x_test.index)
    
    return df_x_test

def apply_nominal_encoder(nominal_encoder, df_x_test):
    """
    Apply nominal encoding (current is Leave-One-Out) for a given list of columns. 
    For more information check the link below:
    https://contrib.scikit-learn.org/category_encoders/leaveoneout.html
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    Category Encoders:
        from category_encoders import LeaveOneOutEncoder
    
    Parameters
    ----------
    nominal_encoder: category_encoders.leave_one_out.LeaveOneOutEncoder
        Trained encoder. Saved in './transformadores_scalers/'
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
            
    Returns
    -------
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (testing set).
    """
    
    # Apply the transformation in train and test data
    df_x_test = nominal_encoder.transform(df_x_test)
    
    return df_x_test

def apply_scalers(df_x_test, continuous_scalers):
    """
    Apply Standard Scaler from scikit-learn for all numerical columns. 
    For more information check the link below:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    Category Encoders:
        from sklearn.preprocessing import StandardScaler
    
    Parameters
    ----------
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
    continuous_scalers : dict
        Dictionary containing column names and their respective scalers, where
        continuous_scalers = {'column_name' : scaler_object}
        
        Ex: 
            continuous_scalers = {
                'num_idade' : standard_scaler_num_idade,
                'num_tempo_casa' : standard_scaler_num_tempo_casa,
                'vlr_salario' : standard_scaler_vlr_salario,
                'qtd_jornada' : standard_scaler_qtd_jornada,
                'TotalMesesDesdeUltimoMeritoPromocao' : standard_scaler_TotalMesesDesdeUltimoMeritoPromocao,
                'Custo_Benef_Mes_Anterior' : standard_scaler_Custo_Benef_Mes_Anterior,
                'Num_Benef_Mes_Anterior' : standard_scaler_Num_Benef_Mes_Anterior
            }
        
    Returns
    -------
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns scaled (testing set).
    """
    
    # Train and transform each columns separately
    for column_name, scaler_object in continuous_scalers.items():

        # Apply the scaling in train and test data
        df_x_test[column_name] = scaler_object.transform(df_x_test[column_name].array.reshape(-1,1))
        
    return df_x_test

def filter_dataframe(df): # Função não utilizada
    """
    Adds a UI on top of a dataframe to let users filter columns.
    For more information check the link below:
    https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/

    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
        from pandas.api.types import (
            is_categorical_dtype,
            is_datetime64_any_dtype,
            is_numeric_dtype,
            is_object_dtype,
        )
    Streamlit:
        import streamlit as st
        import streamlit.components.v1 as components

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Pandas dataframe for filtering.

    Returns
    -------
    df : pandas.core.frame.DataFrame 
        Filtered dataframe
    """

    modify = st.checkbox('Adiciona filtros')

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect('Filtrar dataframe para', df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f'Valores para {column}',
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f'Valores para {column}',
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f'Valores para {column}',
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f'Substring ou regex para {column}',
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def shap_waterfall_plot(model, df_x_train, df_x_test, index):
    """
    Shows local feature importance plot (waterfall type) based on shapley values. 
    For more information check the links below:
    https://towardsdatascience.com/introduction-to-shap-values-and-their-application-in-machine-learning-8003718e6827
    https://medium.com/swlh/push-the-limits-of-explainability-an-ultimate-guide-to-shap-library-a110af566a02
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    Matplotlib:
        import matplotlib.pyplot as plt
    SHAP:
        import shap
    
    Parameters
    ----------
    model : sklearn model object
        Sklearn model. Varies according to the instantiated object.
    df_x_train : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for training. 
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
    index: int
        Index of the observation in df_x_test.
        
    Returns
    -------
    None
    """
    
    # Display plots in jupyter notebook
    shap.initjs()
    
    # Fit TreeExplainer
    explainer = shap.TreeExplainer(model, df_x_train, feature_names=df_x_train.columns.tolist(), model_output='probability')
    
    # Compute shapley values
    shap_values = explainer(df_x_test)
    
    # Build a the feature importance
    shap.waterfall_plot(shap_values[index], max_display=len(df_x_test.columns))
    st.pyplot()

    return None

def to_excel(df):
    """
    Process data to be saved in a .xslx file for download.
    For more information check the links below:
    https://stackoverflow.com/questions/67564627/how-to-download-excel-file-in-python-and-streamlit
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    IO:
        from io import BytesIO
    Pyxlsb
        from pyxlsb import open_workbook as open_xlsb
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for training. 
            
    Returns
    -------
    processed_data : bytes
        Data for .xslx file. 
    """

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Colaboradores')
    workbook = writer.book
    worksheet = writer.sheets['Colaboradores']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()

    return processed_data

# -------------------------------- Load databases --------------------------------

with open('./database/dados_pre_processados_train.pkl', mode = 'rb') as file:
    X_train, y_train = pickle.load(file)

with open('./database/dados_raw_prod.pkl', mode = 'rb') as file:
    X_train_raw = pickle.load(file)

with open('./database/dados_preprocess_prod.pkl', mode = 'rb') as file:
    X_test = pickle.load(file)

# -------------------------- Transformers and scalers ----------------------------

with open('./encoders_scalers/ordinal_encoder_train.pkl', mode='rb') as file:
    ordinal_encoder = pickle.load(file)

with open('./encoders_scalers/leave_one_out_encoder_train.pkl', mode='rb') as file:
    leave_one_out_encoder = pickle.load(file)

with open('./encoders_scalers/standard_scaler_Custo_Benef_Mes_Anterior_train.pkl', mode='rb') as file:
    standard_scaler_Custo_Benef_Mes_Anterior = pickle.load(file)

with open('./encoders_scalers/standard_scaler_Num_Benef_Mes_Anterior_train.pkl', mode='rb') as file:
    standard_scaler_Num_Benef_Mes_Anterior = pickle.load(file)

with open('./encoders_scalers/standard_scaler_TotalMesesDesdeUltimoMeritoPromocao_train.pkl', mode='rb') as file:
    standard_scaler_TotalMesesDesdeUltimoMeritoPromocao = pickle.load(file)

with open('./encoders_scalers/standard_scaler_num_idade_train.pkl', mode='rb') as file:
    standard_scaler_num_idade = pickle.load(file)

with open('./encoders_scalers/standard_scaler_num_tempo_casa_train.pkl', mode='rb') as file:
    standard_scaler_num_tempo_casa = pickle.load(file)

with open('./encoders_scalers/standard_scaler_qtd_jornada_train.pkl', mode='rb') as file:
    standard_scaler_qtd_jornada = pickle.load(file)

with open('./encoders_scalers/standard_scaler_vlr_salario_train.pkl', mode='rb') as file:
    standard_scaler_vlr_salario = pickle.load(file)

# -------------------------------- Trained models --------------------------------

with open('./trained_models/rfe_weighted_accuracy_train.pkl', mode='rb') as file:
    rfe = pickle.load(file)

with open('./trained_models/lgbmclassifier_weighted_accuracy_train.pkl', mode='rb') as file:
    model = pickle.load(file)

# -------------------------------- Model metrics ---------------------------------

with open('./model_metrics/lgbmclassifier_accuracy_optroc_scores.pkl', mode='rb') as file:
    metrics = pickle.load(file)

# ----------------------- Config YAML and hash passwords -------------------------

with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

# Authenticator for login page
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

#hashed_passwords = stauth.Hasher(['148536']).generate()
#print(hashed_passwords)

# ------------------------------------- Main -------------------------------------

def main():

    name, authentication_status, username = authenticator.login('Login', 'main')

    if st.session_state['authentication_status']:
        
        logo = Image.open('./images/cogna_logo.png')
        st.image(logo)

        st.write('')
        st.write('')
        st.write('')

        # Greeting and logout button
        st.write(f"Olá {st.session_state['name']}!")

        authenticator.logout('Logout', 'main')
        
        # Space between Logout button and page content
        st.write('')
        st.write('')
        st.write('')

        probabilistic_threshold = np.around(
            a = metrics.iloc[0, metrics.columns.get_loc('prob_treshold')],
            decimals = 3
        )

        with st.sidebar:
            choose = option_menu(
                'Modelo Preditivo de Rescisão Voluntária',
                ['Home', 'Predições Batch', 'Predições Online', 'Informações'],
                icons=['house', 'files', 'globe', 'question-lg'],
                menu_icon='app-indicator',
                default_index=0,
                styles={
                    'container': {'padding': '5!important', 'background-color': '#ADD8E6'},
                    'icon': {'color': 'purple', 'font-size': '20px'}, 
                    'nav-link': {'font-size': '16px', 'text-align': 'left', 'margin':'0px', '--hover-color': '#8A8AFF'},
                    'nav-link-selected': {'background-color': '#8A8AFF'},
                }
            )
        # App version
        st.sidebar.info('Versão do App: v1.0.0')
        st.sidebar.info('Desenvolvido por People & Analytics')
        st.sidebar.info('Feedback: peopleanalytics@cogna.com.br')

        if choose == 'Home':
            st.header('Modelo preditivo de rescisão vonluntária')
            st.write('')
            st.subheader('Sobre:')
            st.markdown(
                '<div style="text-align: justify;"> \
                Este modelo (piloto) foi desenvolvido para classificar colaboradores que possuem propensão de sair voluntariamente da empresa (rescisão voluntária). \
                Esta ferramenta tem como objetivo auxiliar \
                a tomada de decisão do(a) gestor(a) alinhada principalmente com a experiência e conhecimento do mesmo(a) sobre sua equipe. Essa iniciativa \
                tem como objetivo reduzir a rescisão voluntária a longo prazo, e contribuir para a permanência de talentos na empresa. Existem duas \
                funcionalidades neste aplicativo: "Predições Batch" e "Predições Online". ',
                unsafe_allow_html=True
            )
            st.write('')
            st.subheader('Predições Batch:')
            st.markdown(
                '<div style="text-align: justify;"> \
                Disponibiliza uma base de dados de cada colaborador(a). Nesta \
                base encontram-se as características avaliadas pelo modelo, assim como a classificação desiginada entre 1 (saída voluntária) \
                e  0 (permanência na empresa). Além da classificação, cada colaborador(a) possue um atributo de probabilidade de rescisão voluntária, também caculado \
                pelo modelo. A classificação é feita com base nesta probabilidade. Por exemplo, um colaborador com uma probabilidade de 60% de chance de rescindir o \
                contrato de forma voluntária será classificado como 1 (rescisão voluntária), onde o critério para classificação pode ser: 0 (permanência) se menor que \
                50% e 1 (saída) se maior, ou igual, a 50%.',
                unsafe_allow_html=True
            )
            st.write('')
            st.subheader('Predições Online:')
            st.markdown(
                '<div style="text-align: justify;"> \
                É uma ferramenta onde o gestor pode prencher manualmente as características \
                do colaborador, em seguida o modelo computa em tempo real qual a classificação e probabilidade associada para o cenário de rescisão voluntária.',
                unsafe_allow_html=True
            )

        elif choose == 'Predições Batch':

            st.subheader('Dados e predições dos colaboradores')

            X_test_result = X_train_raw.copy()
            X_test_result['Probabilidade'] = np.around(
                a = model.predict_proba(X_test.loc[:, rfe.support_])[:,1]*100,
                decimals = 2
            )
            
            # Probability prediction fo the minority class (1)
            y_pred = model.predict_proba(X_test.loc[:, rfe.support_])[:,1]

            # Probabilistic threshold adjustment
            y_pred_tuned_thresh = np.where(y_pred >= probabilistic_threshold, 1, 0)

            # Assigning class
            X_test_result['Classificação'] = y_pred_tuned_thresh
           
            # Columns order (MUST BE THE SAME FOR THE TRAINED MODEL!)
            X_test_result = X_test_result[[
                'Classificação',
                'Probabilidade',
                'nom_funcao',
                'nom_grupo_cargos',
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
                'Num_Benef_Mes_Anterior'
            ]]
            
            # Business rules for accessing information in batch prediction
            if (st.session_state['username'] == 'bruno.chimetta') or (st.session_state['username'] == 'elson.junior'):
                X_test_result = X_test_result
            elif st.session_state['username'] == 'fabio.lacerda':
                X_test_result = X_test_result[X_test_result['nom_vp'] == 'VP GENTE E CULTURA']
            elif (st.session_state['username'] == 'anderson.hass') or (st.session_state['username'] == 'daniela.ono') or (st.session_state['username'] == 'andreia.moretti') or (st.session_state['username'] == 'rita.silva'):
                X_test_result = X_test_result[X_test_result['nom_empresa'] == 'SOMOS']

            # Enables filters and customization on the dataframe
            gb = GridOptionsBuilder.from_dataframe(X_test_result)
            gb.configure_pagination(paginationAutoPageSize=True) # Add pagination
            gb.configure_side_bar() # Add a sidebar
            gb.configure_selection('multiple', use_checkbox=False, groupSelectsChildren='Group checkbox select children') #Enable multi-row selection with use_checkbox = True
            #gb.configure_column('nom_funcao', headerCheckboxSelection = True) # Enable select box on the column label to select all entries
            #gb.configure_column('vlr_salario', hide = True) # Hide an specific column
            gridOptions = gb.build()

            grid_response = AgGrid(
                X_test_result,
                gridOptions=gridOptions,
                data_return_mode='AS_INPUT', 
                update_mode='MODEL_CHANGED', 
                fit_columns_on_grid_load=False,
                theme='streamlit', #Add theme color to the table
                enable_enterprise_modules=True,
                height=350, 
                width='100%'#,
                #reload_data=True
            )

            selected = grid_response['selected_rows'] # Select rows dynamically
            observation = pd.DataFrame(selected) # Pass the selected rows to a new dataframe

            if not observation.empty and observation.shape[0] == 1:

                observation.drop(['_selectedRowNodeInfo', 'Probabilidade', 'Classificação'], axis=1, inplace=True)

                if st.button('Detalhes'):
                
                    # Missing data inputation
                    observation['nom_grupo_cargos'] = observation['nom_grupo_cargos'].fillna('NAO INFORMADO')
                    observation['Custo_Benef_Mes_Anterior'] = observation['Custo_Benef_Mes_Anterior'].fillna(round(X_train_raw['Custo_Benef_Mes_Anterior'].median(), 2))
                    observation['Num_Benef_Mes_Anterior'] = observation['Num_Benef_Mes_Anterior'].fillna(round(X_train_raw['Num_Benef_Mes_Anterior'].median(), 2))
                    
                    
                    ordinal_columns = [
                        'nom_grupo_cargos',
                        'nom_grau_instrucao',
                        'num_nivel_hierarquico',
                        'nom_posicao'
                    ]

                    # Apply the transformation to the ordinal independent variables
                    observation = apply_ordinal_encoder(ordinal_encoder, observation, ordinal_columns)

                    nominal_columns = [
                        'nom_empresa',
                        'nom_vp',
                        'nom_sexo',
                        'nom_raca_cor',
                        'dsc_modelo_trabalho',
                        'nom_estado_civil',
                        'nom_funcao',
                        'nom_dir_sr',
                        'nom_dir',
                        'nom_ger_sr',
                        'nom_ger'
                    ]

                    # Apply the transformation to the nominal independent variables
                    observation = apply_nominal_encoder(leave_one_out_encoder, observation)

                    continuous_scalers = {
                        'num_idade' : standard_scaler_num_idade,
                        'num_tempo_casa' : standard_scaler_num_tempo_casa,
                        'vlr_salario' : standard_scaler_vlr_salario,
                        'qtd_jornada' : standard_scaler_qtd_jornada,
                        'TotalMesesDesdeUltimoMeritoPromocao' : standard_scaler_TotalMesesDesdeUltimoMeritoPromocao,
                        'Custo_Benef_Mes_Anterior' : standard_scaler_Custo_Benef_Mes_Anterior,
                        'Num_Benef_Mes_Anterior' : standard_scaler_Num_Benef_Mes_Anterior
                    }

                    # Applies the scaling of quantitative independent variables
                    observation = apply_scalers(observation, continuous_scalers)

                    # Predicting the probability of an observation
                    probability = model.predict_proba(observation.loc[:, rfe.support_])[:,1][0]

                    # Probabilistic threshold adjustment for classification
                    classification = np.where(probability >= probabilistic_threshold, 1, 0)
                    
                    # For better visualization
                    probability = probability*100
                    
                    if classification == 1:
                        st.error(f'Rescisão voluntária: Sim')
                        st.error(f'Probabilidade de saída: {probability:.2f}% ')
                        shap_waterfall_plot(model, X_train.loc[:, rfe.support_], observation.loc[:, rfe.support_], 0)
                        st.write('E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                        st.write('f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')
                    else:
                        st.success(f'Rescisão voluntária: Não')
                        st.success(f'Probabilidade de saída: {probability:.2f}%')
                        shap_waterfall_plot(model, X_train.loc[:, rfe.support_], observation.loc[:, rfe.support_], 0)
                        st.write('E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                        st.write('f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')

            # Block for download functionality
            #elif not observation.empty and observation.shape[0] > 1:
            #
            #    st.subheader('Dados consolidados após filtros')
            #
            #    observation.drop(['_selectedRowNodeInfo'], axis=1, inplace=True)
            #    st.dataframe(observation)
            #
            #    df_xlsx = to_excel(observation)
            #    st.download_button(
            #        label='Download', 
            #        data=df_xlsx, 
            #        file_name= 'Rescisão voluntária - Colaboradores.xlsx'
            #    )

            else:

                st.warning('Selecione um colaborador.')

        elif choose == 'Predições Online':

            # Missing data inputation
            X_train_raw['nom_grupo_cargos'] = X_train_raw['nom_grupo_cargos'].fillna('NAO INFORMADO')

            st.subheader('Características do colaborador(a)')

            nom_funcao = st.selectbox(
                'Função:',
                X_train_raw.loc[:, 'nom_funcao'].unique().tolist()
            )

            nom_grupo_cargos = st.selectbox(
                'Grupo/Cargos:', 
                X_train_raw.loc[:, 'nom_grupo_cargos'].unique().tolist()
            )

            nom_empresa = st.selectbox(
                'Empresa:', 
                X_train_raw.loc[:, 'nom_empresa'].unique().tolist()
            )

            nom_vp = st.selectbox(
                'Nome VP:',
                X_train_raw.loc[:, 'nom_vp'].unique().tolist()
            )

            nom_dir_sr = st.selectbox(
                'Diretor(a) Sr.:', 
                X_train_raw.loc[:, 'nom_dir_sr'].unique().tolist()
            )

            nom_dir = st.selectbox(
                'Diretor:', 
                X_train_raw.loc[:, 'nom_dir'].unique().tolist()
            )

            nom_ger_sr = st.selectbox(
                'Gerente Sr.:', 
                X_train_raw.loc[:, 'nom_ger_sr'].unique().tolist()
            )

            nom_ger = st.selectbox(
                'Gerente:', 
                X_train_raw.loc[:, 'nom_ger'].unique().tolist()
            )

            num_idade = st.number_input(
                'Idade (em anos):', 
                min_value=int(X_train_raw['num_idade'].min()), 
                max_value=int(X_train_raw['num_idade'].max()), 
                value=int(X_train_raw['num_idade'].median())
            )

            num_tempo_casa = st.number_input(
                'Tempo de casa (em meses):', 
                min_value=int(X_train_raw['num_tempo_casa'].min()), 
                max_value=int(X_train_raw['num_tempo_casa'].max()), 
                value=int(X_train_raw['num_tempo_casa'].median())
            )

            # Verificar remoção
            nom_sexo = st.selectbox(
                'Gênero:', 
                X_train_raw.loc[:, 'nom_sexo'].unique().tolist()
            )
             
            num_nivel_hierarquico = st.selectbox(
                'Nível hierárquico:', 
                X_train_raw.loc[:, 'num_nivel_hierarquico'].unique().tolist()
            )

            nom_posicao = st.selectbox(
                'Posição:', 
                X_train_raw.loc[:, 'nom_posicao'].unique().tolist()
            )

            nom_raca_cor = st.selectbox(
                'Etnia/Cor:', 
                X_train_raw.loc[:, 'nom_raca_cor'].unique().tolist()
            )

            dsc_modelo_trabalho = st.selectbox(
                'Modelo de trabalho:', 
                X_train_raw.loc[:, 'dsc_modelo_trabalho'].unique().tolist()
            )

            nom_estado_civil = st.selectbox(
                'Estado civil:', 
                X_train_raw.loc[:, 'nom_estado_civil'].unique().tolist()
            )

            nom_grau_instrucao = st.selectbox(
                'Grau de instrução:', 
                X_train_raw.loc[:, 'nom_grau_instrucao'].unique().tolist()
            )

            vlr_salario = st.number_input(
                'Salário (em reais):', 
                min_value=float(X_train_raw['vlr_salario'].min()), 
                max_value=200000.00, 
                value=float(X_train_raw['vlr_salario'].median())
            )

            qtd_jornada = st.number_input(
                'Jornada (definir):', 
                min_value=int(X_train_raw['qtd_jornada'].min()), 
                max_value=int(X_train_raw['qtd_jornada'].max()), 
                value=int(X_train_raw['qtd_jornada'].median())
            )

            TotalMesesDesdeUltimoMeritoPromocao = st.number_input(
                'Tempo desde o último mérito/promoção (em meses):', 
                min_value=int(X_train_raw['TotalMesesDesdeUltimoMeritoPromocao'].min()), 
                max_value=int(X_train_raw['TotalMesesDesdeUltimoMeritoPromocao'].max()), 
                value=int(X_train_raw['TotalMesesDesdeUltimoMeritoPromocao'].median())
            )

            Custo_Benef_Mes_Anterior = st.number_input(
                'Custo do benefício no mês anterior (em reais):', 
                min_value=float(X_train_raw['Custo_Benef_Mes_Anterior'].min()), 
                max_value=float(X_train_raw['Custo_Benef_Mes_Anterior'].max()), 
                value=float(X_train_raw['Custo_Benef_Mes_Anterior'].median())
            )

            Num_Benef_Mes_Anterior = st.number_input(
                'Utilização do benefício no mês anterior (quantidade):',
                min_value=int(X_train_raw['Num_Benef_Mes_Anterior'].min()), 
                max_value=int(X_train_raw['Num_Benef_Mes_Anterior'].max()), 
                value=int(X_train_raw['Num_Benef_Mes_Anterior'].median())
            )

            
            input_dict = {
                'nom_funcao' : nom_funcao,
                'nom_grupo_cargos' : nom_grupo_cargos,
                'nom_empresa' : nom_empresa,
                'nom_vp' : nom_vp,
                'nom_dir_sr' : nom_dir_sr,
                'nom_dir' : nom_dir,
                'nom_ger_sr' : nom_ger_sr,
                'nom_ger' : nom_ger,
                'num_idade' : num_idade,
                'num_tempo_casa' : num_tempo_casa,
                'nom_sexo' : nom_sexo,
                'num_nivel_hierarquico' : num_nivel_hierarquico,
                'nom_posicao' : nom_posicao,
                'nom_raca_cor' : nom_raca_cor,
                'dsc_modelo_trabalho' : dsc_modelo_trabalho,
                'nom_estado_civil' : nom_estado_civil,
                'nom_grau_instrucao' : nom_grau_instrucao,
                'vlr_salario' : vlr_salario,
                'qtd_jornada' : qtd_jornada,
                'TotalMesesDesdeUltimoMeritoPromocao' : TotalMesesDesdeUltimoMeritoPromocao,
                'Custo_Benef_Mes_Anterior' : Custo_Benef_Mes_Anterior,
                'Num_Benef_Mes_Anterior' : Num_Benef_Mes_Anterior
            }

            # Predict button for online option
            if st.button('Predizer'):

                observation = pd.DataFrame.from_dict(input_dict, orient='index').T

                ordinal_columns = [
                    'nom_grupo_cargos',
                    'nom_grau_instrucao',
                    'num_nivel_hierarquico',
                    'nom_posicao'
                ]

                # Apply the transformation to the ordinal independent variables
                observation = apply_ordinal_encoder(ordinal_encoder, observation, ordinal_columns)

                nominal_columns = [
                    'nom_empresa',
                    'nom_vp',
                    'nom_sexo',
                    'nom_raca_cor',
                    'dsc_modelo_trabalho',
                    'nom_estado_civil',
                    'nom_funcao',
                    'nom_dir_sr',
                    'nom_dir',
                    'nom_ger_sr',
                    'nom_ger'
                ]

                # Apply the transformation to the nominal independent variables
                observation = apply_nominal_encoder(leave_one_out_encoder, observation)

                continuous_scalers = {
                    'num_idade' : standard_scaler_num_idade,
                    'num_tempo_casa' : standard_scaler_num_tempo_casa,
                    'vlr_salario' : standard_scaler_vlr_salario,
                    'qtd_jornada' : standard_scaler_qtd_jornada,
                    'TotalMesesDesdeUltimoMeritoPromocao' : standard_scaler_TotalMesesDesdeUltimoMeritoPromocao,
                    'Custo_Benef_Mes_Anterior' : standard_scaler_Custo_Benef_Mes_Anterior,
                    'Num_Benef_Mes_Anterior' : standard_scaler_Num_Benef_Mes_Anterior
                }

                # Applies the scaling of quantitative independent variables
                observation = apply_scalers(observation, continuous_scalers)

                # Predicting the probability of an observation
                probability = model.predict_proba(observation.loc[:, rfe.support_])[:,1][0]

                # Probabilistic threshold adjustment for classification
                classification = np.where(probability >= probabilistic_threshold, 1, 0)
                
                # For better visualization
                probability = probability*100
                
                if classification == 1:
                    st.error(f'Rescisão voluntária: Sim')
                    st.error(f'Probabilidade de saída: {probability:.2f}% ')
                    shap_waterfall_plot(model, X_test.loc[:, rfe.support_], observation.loc[:, rfe.support_], 0)
                    st.write('E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                    st.write('f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')
                else:
                    st.success(f'Rescisão voluntária: Não')
                    st.success(f'Probabilidade de saída: {probability:.2f}%')
                    shap_waterfall_plot(model, X_test.loc[:, rfe.support_], observation.loc[:, rfe.support_], 0)
                    st.write('E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                    st.write('f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')

        elif choose == 'Informações':

            
            st.subheader('Métricas do modelo')
            st.info('Framework: [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/)')
            st.info('Período de treino: 08/2022 à 01/2023')
            st.info('Período de validade: 02/2023 à 07/2023')
            st.info('Limiar probabilístico de classificação: {0:.1f}%'.format(probabilistic_threshold*100))
            st.info('Acurácia: {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('accuracy')]*100))
            st.info('Recall: {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('recall')]*100))
            st.info('ROCAUC: {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('rocauc')]*100))
            #st.info('Precisão (classe 1): {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('precision')]*100))
            #st.info('F1 (classe 1): {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('f1')]*100))
            st.info('Erro: {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('error')]*100))
            st.info('Margem de erro: +/-{0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('conf_interval')]*100))

            st.subheader('Catágolo dos dados')
            st.info('nom_funcao: Nome da função do(a) colaborador(a)')
            st.info('nom_grupo_cargos: Nome do grupo de cargos do(a) colaborador(a)')
            st.info('nom_empresa: Nome da empresa do(a) colaborador(a)')
            st.info('nom_vp: Nome do(a) VP do(a) colaborador(a)')
            st.info('nom_dir_sr: Nome do diretor(a) sênior do(a) colaborador(a)')
            st.info('nom_dir: Nome do diretor(a) do(a) colaborador(a)')
            st.info('nom_ger_sr: Nome do gerente sênior do(a) colaborador(a)')
            st.info('nom_ger: Nome do gerente do(a) colaborador(a)')
            st.info('num_idade: Idade do(a) colaborador(a)')
            st.info('num_tempo_casa: Tempo de permanência na empresa (em meses)')
            st.info('nom_sexo: Gênero do do(a) colaborador(a)')
            st.info('num_nivel_hierarquico: Nível hierárquico do(a) colaborador(a)')
            st.info('nom_posicao: Posição do(a) colaborador(a)')
            st.info('nom_raca_cor: Etnia/Cor do(a) colaborador(a)')
            st.info('dsc_modelo_trabalho: Modelo de trabalho do(a) colaborador(a)')
            st.info('nom_estado_civil: Estado civil do(a) colaborador(a)')
            st.info('nom_grau_instrucao: Grau de instrução do(a) colaborador(a)')
            st.info('vlr_salario: Valor do salário do(a) colaborador(a)')
            st.info('qtd_jornada: Jornada de trabalho do(a) colaborador(a) (em horas)')
            st.info('TotalMesesDesdeUltimoMeritoPromocao: Tempo desde o último mérito ou promoção do(a) colaborador(a) (em meses)')
            st.info('Custo_Benef_Mes_Anterior: Custo dos benefícios do(a) colaborador(a) (no período de treino do modelo)')
            st.info('Num_Benef_Mes_Anterior: Número de vezes em que benefícios foram utilizados pelo(a) colaborador(a) (no período de treino do modelo)')

    elif st.session_state['authentication_status'] == False:
        st.error('Username/password incorretos')

    elif st.session_state['authentication_status'] == None:
        st.warning('Por favor, entre com seu username e password')


if __name__ == '__main__':
    main()