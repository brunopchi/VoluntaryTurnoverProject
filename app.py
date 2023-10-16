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

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

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
    encoded_array_test = ordinal_encoder.transform(
        df_x_test.loc[:, ordinal_columns])

    # Build the dataframes preserving original indexes
    df_x_test[ordinal_columns] = pd.DataFrame(
        encoded_array_test, columns=ordinal_encoder.get_feature_names_out(), index=df_x_test.index)

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
        df_x_test[column_name] = scaler_object.transform(
            df_x_test[column_name].array.reshape(-1, 1))

    return df_x_test


def filter_dataframe(df):  # Função não utilizada
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
        to_filter_columns = st.multiselect(
            'Filtrar dataframe para', df.columns)
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
                    user_date_input = tuple(
                        map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f'Substring ou regex para {column}',
                )
                if user_text_input:
                    df = df[df[column].astype(
                        str).str.contains(user_text_input)]

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
    explainer = shap.TreeExplainer(
        model, df_x_train, feature_names=df_x_train.columns.tolist(), model_output='probability')

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


with open('./database/dados_pre_processados.pkl', mode='rb') as file:
    X_train, y_train, X_test, y_test = pickle.load(file)

with open('./database/dados_gridsearchcv.pkl', mode='rb') as file:
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = pickle.load(file)

# with open('./database/dados_pre_processados.pkl', mode='rb') as file:
#    X_test = pickle.load(file)

# -------------------------- Transformers and scalers ----------------------------

with open('./encoders_scalers/ordinal_encoder.pkl', mode='rb') as file:
    ordinal_encoder = pickle.load(file)

with open('./encoders_scalers/leave_one_out_encoder.pkl', mode='rb') as file:
    leave_one_out_encoder = pickle.load(file)

with open('./encoders_scalers/standard_scaler_avg_hrs_month.pkl', mode='rb') as file:
    standard_scaler_avg_hrs_month = pickle.load(file)

with open('./encoders_scalers/standard_scaler_projects.pkl', mode='rb') as file:
    standard_scaler_projects = pickle.load(file)

with open('./encoders_scalers/standard_scaler_review.pkl', mode='rb') as file:
    standard_scaler_review = pickle.load(file)

with open('./encoders_scalers/standard_scaler_satisfaction.pkl', mode='rb') as file:
    standard_scaler_satisfaction = pickle.load(file)

with open('./encoders_scalers/standard_scaler_tenure.pkl', mode='rb') as file:
    standard_scaler_tenure = pickle.load(file)

# -------------------------------- Trained models --------------------------------

with open('./trained_models/rfecv_selection_roc_auc.pkl', mode='rb') as file:
    rfecv_list = pickle.load(file)

with open('./trained_models/lgbmclassifier_weighted_roc_auc.pkl', mode='rb') as file:
    model = pickle.load(file)

# -------------------------------- Model metrics ---------------------------------

with open('./model_metrics/lgbmclassifier_roc_auc_optprecisionrecall_scores.pkl', mode='rb') as file:
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
# print(hashed_passwords)

# ------------------------------------- Main -------------------------------------


def main():

    name, authentication_status, username = authenticator.login(
        'Login', 'main')

    if st.session_state['authentication_status']:

        #logo = Image.open('./images/no_image.png')
        # st.image(logo)

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
            a=metrics.iloc[0, metrics.columns.get_loc('prob_treshold')],
            decimals=3
        )

        with st.sidebar:
            choose = option_menu(
                'Modelo Preditivo de Rescisão Voluntária',
                ['Home', 'Predições Batch', 'Predições Online', 'Informações'],
                icons=['house', 'files', 'globe', 'question-lg'],
                menu_icon='app-indicator',
                default_index=0,
                styles={
                    'container': {'padding': '5!important', 'background-color': '#d5def5'},
                    'icon': {'color': 'purple', 'font-size': '20px'},
                    'nav-link': {'font-size': '16px', 'text-align': 'left', 'margin': '0px', '--hover-color': '#8A8AFF'},
                    'nav-link-selected': {'background-color': '#8A8AFF'},
                }
            )
        # App version
        st.sidebar.info('Versão do App: v1.0.0')
        #st.sidebar.info('Desenvolvido por ')
        st.sidebar.info('Feedback: brunopchimetta@gmail.com')

        if choose == 'Home':
            st.header('Modelo preditivo de rescisão vonluntária')
            st.write('')
            st.subheader('Sobre:')
            st.markdown(
                '<div style="text-align: justify;"> \
                Esta aplicação foi desenvolvida como parte do projeto de TCC do curso MBA USP/ESALQ em Data Science & Analytics, para classificar colaboradores \
                que possuem propensão de sair voluntariamente da empresa (rescisão voluntária). Esta ferramenta tem como objetivo auxiliar \
                a tomada de decisão do(a) gestor(a) alinhada principalmente com a experiência e conhecimento do mesmo(a) sobre sua equipe. Essa iniciativa \
                tem como objetivo reduzir a rescisão voluntária a longo prazo, e contribuir para a permanência de talentos na empresa. Existem duas \
                funcionalidades neste aplicativo: "Predições Batch" e "Predições Online". ',
                unsafe_allow_html=True
            )
            st.write('')
            st.subheader('Predições Batch:')
            st.markdown(
                '<div style="text-align: justify;"> \
                Disponibiliza uma base de dados com todos os colaboradores. Nesta \
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

            X_test_result = X_test_raw.copy()
            X_test_result['Probabilidade'] = np.around(
                a=model.predict_proba(X_test.loc[:, rfecv_list])[:, 1]*100,
                decimals=2
            )

            # Probability prediction fo the minority class (1)
            y_pred = model.predict_proba(X_test.loc[:, rfecv_list])[:, 1]

            # Probabilistic threshold adjustment
            y_pred_tuned_thresh = np.where(
                y_pred >= probabilistic_threshold, 1, 0)

            # Assigning class
            X_test_result['Classificação'] = y_pred_tuned_thresh

            # Columns order (MUST BE THE SAME FOR THE TRAINED MODEL!)
            X_test_result = X_test_result[[
                'Probabilidade',
                'Classificação',
                'department',
                'promoted',
                'review',
                'projects',
                'salary',
                'tenure',
                'satisfaction',
                'bonus',
                'avg_hrs_month'
            ]]

            # Business rules for accessing information in batch prediction
            if (st.session_state['username'] == 'bruno.chimetta'):
                X_test_result = X_test_result

            # Enables filters and customization on the dataframe
            gb = GridOptionsBuilder.from_dataframe(X_test_result)
            gb.configure_pagination(
                paginationAutoPageSize=True)  # Add pagination
            gb.configure_side_bar()  # Add a sidebar
            # Enable multi-row selection with use_checkbox = True
            gb.configure_selection('multiple', use_checkbox=True,
                                   groupSelectsChildren='Group checkbox select children')
            # Enable select box on the column label to select all entries
            gb.configure_column('Probabilidade', headerCheckboxSelection=True)
            # gb.configure_column('vlr_salario', hide = True) # Hide an specific column
            gridOptions = gb.build()

            grid_response = AgGrid(
                X_test_result,
                gridOptions=gridOptions,
                data_return_mode='AS_INPUT',
                update_mode='MODEL_CHANGED',
                fit_columns_on_grid_load=False,
                theme='streamlit',  # Add theme color to the table
                enable_enterprise_modules=True,
                height=350,
                width='100%'  # ,
                # reload_data=True
            )

            # Select rows dynamically
            selected = grid_response['selected_rows']
            # Pass the selected rows to a new dataframe
            observation = pd.DataFrame(selected)

            if not observation.empty and observation.shape[0] == 1:

                observation.drop(
                    ['_selectedRowNodeInfo', 'Probabilidade', 'Classificação'], axis=1, inplace=True)

                if st.button('Detalhes'):

                    ordinal_columns = [
                        'salary'
                    ]

                    # Apply the transformation to the ordinal independent variables
                    observation = apply_ordinal_encoder(
                        ordinal_encoder, observation, ordinal_columns)

                    nominal_columns = [
                        'department'
                    ]

                    # Apply the transformation to the nominal independent variables
                    observation = apply_nominal_encoder(
                        leave_one_out_encoder, observation)

                    continuous_scalers = {
                        'avg_hrs_month': standard_scaler_avg_hrs_month,
                        'projects': standard_scaler_projects,
                        'review': standard_scaler_review,
                        'satisfaction': standard_scaler_satisfaction,
                        'tenure': standard_scaler_tenure
                    }

                    # Applies the scaling of quantitative independent variables
                    observation = apply_scalers(
                        observation, continuous_scalers)

                    # Predicting the probability of an observation
                    probability = model.predict_proba(
                        observation.loc[:, rfecv_list])[:, 1][0]

                    # Probabilistic threshold adjustment for classification
                    classification = np.where(
                        probability >= probabilistic_threshold, 1, 0)

                    # For better visualization
                    probability = probability*100

                    if classification == 1:
                        st.error(f'Rescisão voluntária: Sim')
                        st.error(
                            f'Probabilidade de saída: {probability:.2f}% ')
                        shap_waterfall_plot(
                            model, X_train.loc[:, rfecv_list], observation.loc[:, rfecv_list], 0)
                        st.write(
                            'E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                        st.write(
                            'f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')
                    else:
                        st.success(f'Rescisão voluntária: Não')
                        st.success(
                            f'Probabilidade de saída: {probability:.2f}%')
                        shap_waterfall_plot(
                            model, X_train.loc[:, rfecv_list], observation.loc[:, rfecv_list], 0)
                        st.write(
                            'E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                        st.write(
                            'f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')

            # Block for download functionality
            elif not observation.empty and observation.shape[0] > 1:

                st.subheader('Dados consolidados após filtros')

                observation.drop(['_selectedRowNodeInfo'],
                                 axis=1, inplace=True)
                st.dataframe(observation)

                df_xlsx = to_excel(observation)
                st.download_button(
                    label='Download',
                    data=df_xlsx,
                    file_name='Rescisão voluntária - Colaboradores.xlsx'
                )

            else:

                st.warning('Selecione um colaborador.')

        elif choose == 'Predições Online':

            st.subheader('Características do colaborador(a)')

            department = st.selectbox(
                'Departamento:',
                X_train_raw.loc[:, 'department'].unique().tolist()
            )

            promoted = st.selectbox(
                'Promovido (últimos 24 meses):',
                X_train_raw.loc[:, 'promoted'].unique().tolist()
            )

            review = st.number_input(
                'Revisão (última avaliação):',
                min_value=int(X_train_raw['review'].min()),
                max_value=int(X_train_raw['review'].max()),
                value=int(X_train_raw['review'].median())
            )

            projects = st.number_input(
                'Projetos:',
                min_value=int(X_train_raw['projects'].min()),
                max_value=int(X_train_raw['projects'].max()),
                value=int(X_train_raw['projects'].median())
            )

            salary = st.selectbox(
                'Salário:',
                X_train_raw.loc[:, 'salary'].unique().tolist()
            )

            tenure = st.number_input(
                'Permanência:',
                min_value=int(X_train_raw['tenure'].min()),
                max_value=int(X_train_raw['tenure'].max()),
                value=int(X_train_raw['tenure'].median())
            )

            satisfaction = st.number_input(
                'Satisfação:',
                min_value=int(X_train_raw['satisfaction'].min()),
                max_value=int(X_train_raw['satisfaction'].max()),
                value=int(X_train_raw['satisfaction'].median())
            )

            bonus = st.number_input(
                'Bonus (últimos 24 meses):',
                min_value=int(X_train_raw['bonus'].min()),
                max_value=int(X_train_raw['bonus'].max()),
                value=int(X_train_raw['bonus'].median())
            )

            avg_hrs_month = st.number_input(
                'Horas trabalhadas por mês (média):',
                min_value=int(X_train_raw['avg_hrs_month'].min()),
                max_value=int(X_train_raw['avg_hrs_month'].max()),
                value=int(X_train_raw['avg_hrs_month'].median())
            )

            input_dict = {
                'department': department,
                'promoted': promoted,
                'review': review,
                'projects': projects,
                'salary': salary,
                'tenure': tenure,
                'satisfaction': satisfaction,
                'bonus': bonus,
                'avg_hrs_month': avg_hrs_month,
            }

            # Predict button for online option
            if st.button('Predizer'):

                observation = pd.DataFrame.from_dict(
                    input_dict, orient='index').T

                ordinal_columns = [
                    'salary'
                ]

                # Apply the transformation to the ordinal independent variables
                observation = apply_ordinal_encoder(
                    ordinal_encoder, observation, ordinal_columns)

                nominal_columns = [
                    'department'
                ]

                # Apply the transformation to the nominal independent variables
                observation = apply_nominal_encoder(
                    leave_one_out_encoder, observation)

                continuous_scalers = {
                    'avg_hrs_month': standard_scaler_avg_hrs_month,
                    'projects': standard_scaler_projects,
                    'review': standard_scaler_review,
                    'satisfaction': standard_scaler_satisfaction,
                    'tenure': standard_scaler_tenure
                }

                # Applies the scaling of quantitative independent variables
                observation = apply_scalers(observation, continuous_scalers)

                # Predicting the probability of an observation
                probability = model.predict_proba(
                    observation.loc[:, rfecv_list])[:, 1][0]

                # Probabilistic threshold adjustment for classification
                classification = np.where(
                    probability >= probabilistic_threshold, 1, 0)

                # For better visualization
                probability = probability*100

                if classification == 1:
                    st.error(f'Rescisão voluntária: Sim')
                    st.error(f'Probabilidade de saída: {probability:.2f}% ')
                    shap_waterfall_plot(
                        model, X_test.loc[:, rfecv_list], observation.loc[:, rfecv_list], 0)
                    st.write(
                        'E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                    st.write(
                        'f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')
                else:
                    st.success(f'Rescisão voluntária: Não')
                    st.success(f'Probabilidade de saída: {probability:.2f}%')
                    shap_waterfall_plot(
                        model, X_test.loc[:, rfecv_list], observation.loc[:, rfecv_list], 0)
                    st.write(
                        'E[f(x)]: probabilidade (entre 0 e 1) de rescisão voluntária caso nenhuma informação fosse fornecida')
                    st.write(
                        'f(x): probabilidade (entre 0 e 1) de rescisão voluntária com base nas informações fornecidas')

        elif choose == 'Informações':

            st.subheader('Métricas do modelo')
            st.info(
                'Framework: [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/)')
            st.info('Limiar probabilístico de classificação: {0:.1f}%'.format(
                probabilistic_threshold*100))
            st.info('Acurácia: {0:.2f}%'.format(
                metrics.iloc[0, metrics.columns.get_loc('accuracy')]*100))
            st.info('Recall: {0:.2f}%'.format(
                metrics.iloc[0, metrics.columns.get_loc('recall')]*100))
            st.info('ROCAUC: {0:.2f}%'.format(
                metrics.iloc[0, metrics.columns.get_loc('rocauc')]*100))
            #st.info('Precisão (classe 1): {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('precision')]*100))
            #st.info('F1 (classe 1): {0:.2f}%'.format(metrics.iloc[0, metrics.columns.get_loc('f1')]*100))
            st.info('Erro: {0:.2f}%'.format(
                metrics.iloc[0, metrics.columns.get_loc('error')]*100))
            st.info('Margem de erro: +/-{0:.2f}%'.format(
                metrics.iloc[0, metrics.columns.get_loc('conf_interval')]*100))

            st.subheader('Catágolo dos dados')
            st.info(
                'Departamento (department): o departamento ao qual o(a) colaborador(a) pertence.')
            st.info(
                'Promovido (promoted): 1 se o(a) colaborador(a) foi promovido(a) nos 24 meses anteriores, 0 caso contrário.')
            st.info(
                'Revisão (review): a pontuação composta que o(a) colaborador(a) recebeu em sua última avaliação.')
            st.info(
                'Projetos (projects): quantos projetos o(a) colaborador(a) está envolvido(a).')
            st.info(
                'Salário (salary): o salário do caloborador(a) em níveis: baixo (low), médio(medium), alto(high).')
            st.info(
                'Permanência (tenure): quantos anos o(a) colaborador(a) está, ou ficou, na empresa.')
            st.info(
                'Satisfação (satisfaction): uma medida de satisfação do(a) colaborador(a), a partir de questionários.')
            st.info(
                'Bônus (bonus): 1 se o(a) colaborador(a) recebeu um bonus nos 24 meses anteriores, 0 caso contrário.')
            st.info(
                'Média de horas por mês (avg_hrs_month): a média de horas que o(a) colaborador(a) trabalhou num mês')
            st.info(
                'Rescisão (left): "sim" se o(a) colaborador(a) saiu da empresa, "não" caso contrário.')
            st.info(
                'Fonte dos dados: [Kaggle](https://www.kaggle.com/datasets/marikastewart/employee-turnover)')

    elif st.session_state['authentication_status'] == False:
        st.error('Username/password incorretos')

    elif st.session_state['authentication_status'] == None:
        st.warning('Por favor, entre com seu username e password')


if __name__ == '__main__':
    main()
