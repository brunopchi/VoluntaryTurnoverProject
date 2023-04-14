# Básico
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt

# Pipelines, Transformadores e Modelos
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# -------------------------------- Functions --------------------------------

def easy_target_variable_encoder(df, old_target_variable, target_category, new_target_variable=None):
    """
    Function suited for target variables with only two categories (classes). 
    Encode the given target_variable as 1 and 0. Caution: even missing values 
    will be encoded as 0.
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe that contains all data.
    old_target_variable: str
        Column name of the targe variable (dependent variable).
    target_category: str
        Category (class), presented in the target variable, that will be
        encoded as 1. The other category will be encoded as 0, even
        missing values.
    new_target_variable: str
        (Optional) New column name of the targe variable (dependent variable).
        Default value is None.
        
    Returns
    -------
    df : pandas.core.frame.DataFrame
        Pandas dataframe with dependent variable (target variable) encoded
        as binary column.
    """
    
    # Encode target_variable for a binary classification problem
    df[old_target_variable] = df[old_target_variable].apply(lambda x: 1 if x == target_category else 0)
        
    if new_target_variable == None:
        return df
    
    else:
        # Rename target_variable
        df.rename(columns={old_target_variable : new_target_variable} , inplace=True)

        return df


def easy_target_variable_organizer(df, target_variable):
    """
    Function that shifts the target_variable column to the last postion of the 
    dataframe and split independent variables from the target variable (dependent). 
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
        
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Dataframe that contains all data.
    target_variable: str
        Column name of the target variable (dependent).
    target_category:
        Category (class), presented in the target variable, that will be
        encoded as 1. The other category will be encoded as 0, even
        missing values.
        
    Returns
    -------
    df_x : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables.
    df_y : pandas.core.series.Series
        Pandas series containing the target variable (dependent).
    """
    
    # Move target_variable column 'SaidaVoluntaria' to the last position 
    df = df.reindex(columns = [col for col in df.columns if col != target_variable] + [target_variable])
    
    # Independent variables
    df_x = df.iloc[:, 0:df.shape[1]-1]

    # Dependent variable (target_variable)
    df_y = df.iloc[:, df.shape[1]-1]

    return df_x, df_y


def easy_ordinal_encoder(df_x_train, ordinal_columns, ordinal_categories):
    """
    Apply Ordinal Encode from scikit-learn for a given list of columns and categories. 
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
    Pickle:
        import pickle
    
    Parameters
    ----------
    df_x_train : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for training. 
    ordinal_columns : list
        List containing the names of all columns to be transformed. Preferably 
        composed of ordinal variables. 
    ordinal_categories : list
        List of lists containing all categories in each columns. The order of
        each category inside the list dictates her order in the encoding process.
        
        Ex: weather_list = ['cold', 'warm', hot] will be encoded as
            weather_list = [0, 1, 2].
        
    Returns
    -------
    df_x_train: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (training set).
    encoder: sklearn.preprocessing._encoders.OrdinalEncoder
        Trained encoder. Saved in './encoders_scalers/'
    """
    
    # Instantiation
    encoder = OrdinalEncoder(
        categories= ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )

    # Fit using train data
    encoder.fit(df_x_train[ordinal_columns])

    # Apply the transformation in train data
    encoded_array_train = encoder.transform(df_x_train.loc[:,ordinal_columns]) 
   
    # Build the dataframes preserving original indexes
    df_x_train[ordinal_columns] = pd.DataFrame(encoded_array_train,columns=encoder.get_feature_names_out(), index=df_x_train.index)
        
    with open('./encoders_scalers/ordinal_encoder_train.pkl', mode='wb') as file:
        pickle.dump(encoder, file)
    
    return df_x_train


def easy_leave_one_out_encoder(df_x_train, df_y_train, method='auto', nominal_columns=[]):
    """
    Apply Leave-One-Out encoder from scikit-learn for a given list of columns. 
    For more information check the link below:
    https://contrib.scikit-learn.org/category_encoders/leaveoneout.html
    
    Necessary Packages
    ------------------
    Pandas:
        import pandas as pd
    Category Encoders:
        from category_encoders import LeaveOneOutEncoder
    Pickle:
        import pickle
    
    Parameters
    ----------
    df_x_train : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for training. 
    df_y_train : pandas.core.series.Series
        Pandas series containing the dependent variable for training (target variable).
    method : str
        Method for selecting columns: 'declarative' or 'auto'.
            'declarative': give a list os columns names for encoding.
            'auto': all string and categorical columns will be encoded. Default option.
    numerical_columns : list
        List containing all columns names for encoding. Default option is empty [].
        
    Returns
    -------
    df_x_train: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (training set).
    encoder: category_encoders.leave_one_out.LeaveOneOutEncoder
        Trained encoder. Saved in './encoders_scalers/'
    """
    
    if method == 'declarative':
        # Instantiation
        encoder = LeaveOneOutEncoder(cols=nominal_columns, return_df=True)
    else:
       # Instantiation
        encoder = LeaveOneOutEncoder(return_df=True)

    # Fit using train data
    encoder.fit(df_x_train, df_y_train)

    # Apply the transformation in train data
    df_x_train = encoder.transform(df_x_train)
        
    with open('./encoders_scalers/leave_one_out_encoder_train.pkl', mode='wb') as file:
        pickle.dump(encoder, file)
    
    return df_x_train


def easy_standard_scaler(df_x_train, method='auto', numerical_columns=[]):
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
    Pickle:
        import pickle
    
    Parameters
    ----------
    df_x_train : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for training. 
    method : str
        Method for selecting columns: 'declarative' or 'auto'.
            'declarative': give a list os columns names for scaling.
            'auto': scales all numerical columns. Default option.
    numerical_columns : list
        List containing all columns names for scaling. Default option is empty [].
        
    Returns
    -------
    df_x_train: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns scaled (training set).
    scaler: sklearn.preprocessing._data.StandardScaler
        Trained scaler. Saved in './encoders_scalers/'
    """
    
    if method == 'declarative':
        continuous_columns = numerical_columns
    else:
        continuous_columns = list(df_x_train.select_dtypes(include=['int64', 'float64']).columns)

    # Instantiation
    scaler = StandardScaler()
    
    # Train and transform each columns separately
    for continuous_variable in continuous_columns:
        
        # Fit using train data
        scaler.fit(df_x_train[continuous_variable].array.reshape(-1,1))
        
        # Apply the scaling in train data
        df_x_train[continuous_variable] = scaler.transform(df_x_train[continuous_variable].array.reshape(-1,1))
                
        with open(f'./encoders_scalers/standard_scaler_{continuous_variable}_train.pkl', mode='wb') as file:
            pickle.dump(scaler, file)
        
    return df_x_train

# ------------------------- Classification metrics for GridSearchCV -------------------------

# Classifier evaluation metric
evaluation_metric_list = ['accuracy']#['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'neg_log_loss']

for evaluation_metric in evaluation_metric_list:

# -------------------------- Pipeline with output for GridSearchCV --------------------------

    # Import abt
    df = pd.read_csv('./database/abt_train_data.csv')

    # Build target variable
    df = easy_target_variable_encoder(df=df,
                                    old_target_variable='dsc_iniciativa_desligamento',
                                    target_category='Empregado',
                                    new_target_variable='SaidaVoluntaria')

    # Split between dependent and independent variables
    X_train, y_train = easy_target_variable_organizer(df=df, target_variable='SaidaVoluntaria')

    # Save raw data to GridSearchCV
    with open('./database/dados_gridsearchcv_train.pkl', mode= 'wb') as file:
        pickle.dump([X_train, y_train], file)

    # Checkpoint
    print('Raw data saved!')

# --------- Pipeline with output for model training and testing after hyperparameter calibration ---------

    # Missing data handling
    # Train
    X_train['nom_grupo_cargos'] = X_train['nom_grupo_cargos'].fillna('NAO INFORMADO')
    X_train['Custo_Benef_Mes_Anterior'] = X_train['Custo_Benef_Mes_Anterior'].fillna(round(X_train['Custo_Benef_Mes_Anterior'].median(), 2))
    X_train['Num_Benef_Mes_Anterior'] = X_train['Num_Benef_Mes_Anterior'].fillna(round(X_train['Num_Benef_Mes_Anterior'].median(), 2))

    # Ordinal Encoder for ordinal variables
    columns_ordinal = [
        'nom_grupo_cargos', 
        'nom_grau_instrucao', 
        'num_nivel_hierarquico', 
        'nom_posicao'
    ]

    nom_grupo_cargos = [
        'NAO INFORMADO',
        'Tecnico Especialista',
        'Adm/Operacionais',
        'Coord Cursos',
        'Coord Corporativo',
        'Gerentes',
        'Diretores Unidade',
        'Gerentes Sr',
        'Diretores',
        'Vice Presidente',
        'Presidentes',
        'Conselheiros'
    ]

    nom_grau_instrucao = [
        'Analfabeto', 
        'Até o 5º ano incompleto do ensino fundamental', 
        '5º ano completo do ensino fundamental', 
        'Do 6º ao 9º ano do ensino fundamental', 
        'Ensino fundamental completo', 
        'Ensino médio incompleto',
        'Ensino médio completo',
        'Educação superior incompleto',
        'Educação superior completo',
        'Pós Grad. incompleto',
        'Pós Grad. completo',
        'Mestrado incompleto',
        'Mestrado completo',
        'Doutorado incompleto',
        'Doutorado completo',
        'Pós Dout.incompleto',
        'Pós Dout.completo'
    ]

    num_nivel_hierarquico = [
        'N7', 
        'N6', 
        'N5', 
        'N4', 
        'N3', 
        'N2',
        'N1',
        'N0'     
    ]

    nom_posicao = [
        'Operacional', 
        'Lider'
    ]

    categories_ordinal = [
        nom_grupo_cargos, 
        nom_grau_instrucao, 
        num_nivel_hierarquico, 
        nom_posicao
    ]

    X_train = easy_ordinal_encoder(
        df_x_train=X_train,
        ordinal_columns=columns_ordinal,
        ordinal_categories=categories_ordinal
    )

    # Leave-One-Out encoder for nominal variables
    columns_nominal = [
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

    X_train = easy_leave_one_out_encoder(
        df_x_train=X_train,
        df_y_train=y_train,
        method='declarative',
        nominal_columns=columns_nominal
    )

    # Feature scaling
    columns_numerical = [
        'num_idade',
        'num_tempo_casa',
        'vlr_salario',
        'qtd_jornada',
        'TotalMesesDesdeUltimoMeritoPromocao',
        'Custo_Benef_Mes_Anterior',
        'Num_Benef_Mes_Anterior'
    ]

    X_train = easy_standard_scaler(
        df_x_train=X_train,
        method='declarative',
        numerical_columns=columns_numerical
    )


    # Saves train data
    with open('./database/dados_pre_processados_train.pkl', mode= 'wb') as file:
        pickle.dump([X_train, y_train], file)

    # Checkpoint
    print('Pre-processed data saved!')

# -------------- Processing pipeline (gridsearch, training) -------------

    # Input for hyperparameter optimization
    with open('./database/dados_gridsearchcv_train.pkl', mode= 'rb') as file:
        X_train, y_train = pickle.load(file)    

# --------------------- Preprocessing pipeline integrated with GridSearchCV ---------------------

    nom_grupo_cargos = [
        'NAO INFORMADO',
        'Tecnico Especialista',
        'Adm/Operacionais',
        'Coord Cursos',
        'Coord Corporativo',
        'Gerentes',
        'Diretores Unidade',
        'Gerentes Sr',
        'Diretores',
        'Vice Presidente',
        'Presidentes',
        'Conselheiros'
    ]

    nom_grau_instrucao = [
        'Analfabeto', 
        'Até o 5º ano incompleto do ensino fundamental', 
        '5º ano completo do ensino fundamental', 
        'Do 6º ao 9º ano do ensino fundamental', 
        'Ensino fundamental completo', 
        'Ensino médio incompleto',
        'Ensino médio completo',
        'Educação superior incompleto',
        'Educação superior completo',
        'Pós Grad. incompleto',
        'Pós Grad. completo',
        'Mestrado incompleto',
        'Mestrado completo',
        'Doutorado incompleto',
        'Doutorado completo',
        'Pós Dout.incompleto',
        'Pós Dout.completo'
    ]

    num_nivel_hierarquico = [
        'N7', 
        'N6', 
        'N5', 
        'N4', 
        'N3', 
        'N2',
        'N1',
        'N0'     
    ]

    nom_posicao = [
        'Operacional', 
        'Lider'
    ]

    categories_ordinal = [nom_grupo_cargos, nom_grau_instrucao, num_nivel_hierarquico, nom_posicao]

    ordinal_features = [
        'nom_grupo_cargos', 
        'nom_grau_instrucao', 
        'num_nivel_hierarquico', 
        'nom_posicao'
    ]

    ordinal_tranformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NAO INFORMADO')),
            ('oe', OrdinalEncoder(categories= categories_ordinal,
                                handle_unknown='use_encoded_value',
                                unknown_value=np.nan,
                                encoded_missing_value=np.nan))
        ]
    )

    nominal_features_high_cardinality = [
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

    nominal_transformer_high_cardinality = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NAO INFORMADO')),
            ('loo', LeaveOneOutEncoder(return_df=True)),
        ]
    )

    numeric_features = (
        'num_idade',
        'num_tempo_casa',
        'vlr_salario',
        'qtd_jornada',
        'TotalMesesDesdeUltimoMeritoPromocao',
        'Custo_Benef_Mes_Anterior',
        'Num_Benef_Mes_Anterior'
    )

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('nominal_high', nominal_transformer_high_cardinality, nominal_features_high_cardinality),
            ('ordinal', ordinal_tranformer, ordinal_features),
            ('scaler', numeric_transformer, numeric_features)
        ]
    )

# -----------------------------------------------------------------------------------

# ----------------------------- Compute class weightss ------------------------------

    counter_train = Counter(y_train)
    balance_scaler = counter_train[0]/counter_train[1]

# -----------------------------------------------------------------------------------

# ----------------------- Classifier and hyperparameter space -----------------------


    estimators = {
        'lgbmclassifier' : LGBMClassifier()
    }

    params = {
        'lgbmclassifier' : {
            'lgbmclassifier__boosting_type' : ['gbdt', 'dart'],
            'lgbmclassifier__learning_rate' : [0.3, 0.1, 0.01],
            'lgbmclassifier__objective' : ['binary'],
            'lgbmclassifier__n_estimators' : [200, 300, 500, 700],
            'lgbmclassifier__max_depth' : [3, 5, 7, 9, 11],
            'lgbmclassifier__num_leaves' : [3, 5, 7, 9, 11],
            'lgbmclassifier__scale_pos_weight' : [balance_scaler],
            'lgbmclassifier__random_state' : [42]
        }
    }

# -----------------------------------------------------------------------------------

# ---------------------- LGBMClassifier GridSearchCV pipeline -----------------------

    for model_name, model_object in estimators.items():

        # Pipeline instantiation (preprocessing and processing)
        pipe = make_pipeline(
            preprocessor,
            model_object
        )

        # Instantiation of data stratification (preserves the proportion 
        # of classes between training and testing)
        stratkf = StratifiedKFold(n_splits = 10)

        # Instantiation of experiments with GridSearchCV
        grid_search = GridSearchCV(
            estimator = pipe,
            param_grid = params[model_name],
            scoring = evaluation_metric,
            cv = stratkf.split(X_train, y_train),
            return_train_score = True,
            verbose = False
        )

        grid_search.fit(X_train, y_train)

    print(f'GridSearchCV done for {evaluation_metric}!')

    # Loading the pre-processed base for model training
    with open('./database/dados_pre_processados_train.pkl', mode= 'rb') as file:
        X_train, y_train = pickle.load(file)

    # Checkpoint
    print(f'Pre-processed data loaded for {evaluation_metric}!')
    
    # -----------------------------------------------------------------------------------

    # ------------------ Model training with the best hyperparameters -------------------

    # Model instantiation
    lgbm = LGBMClassifier(
        boosting_type= grid_search.best_params_['lgbmclassifier__boosting_type'],
        learning_rate = grid_search.best_params_['lgbmclassifier__learning_rate'],
        objective = grid_search.best_params_['lgbmclassifier__objective'],
        n_estimators = grid_search.best_params_['lgbmclassifier__n_estimators'],
        max_depth = grid_search.best_params_['lgbmclassifier__max_depth'],
        num_leaves = grid_search.best_params_['lgbmclassifier__num_leaves'],
        scale_pos_weight = grid_search.best_params_['lgbmclassifier__scale_pos_weight'],
        random_state = grid_search.best_params_['lgbmclassifier__random_state']
    )

    # Recursive Feature Elimination (RFE) - Best results with 15 features
    number_of_features = 15
    rfe = RFE(estimator=lgbm, n_features_to_select=number_of_features, step=1)

    # Train RFE
    rfe.fit(X_train, y_train)

    # Checkpoint
    print(f'Models LGBM and RFE trained for {evaluation_metric}!')

    # Save trained RFE model
    with open(f'./trained_models/rfe_weighted_{evaluation_metric}_train.pkl', mode= 'wb') as file:
        pickle.dump(rfe, file)

    # Selection of the most relevant features
    X_train = X_train.loc[:, rfe.support_]

    # Train the model
    lgbm.fit(X_train, y_train)

    # Save trained model
    with open(f'./trained_models/lgbmclassifier_weighted_{evaluation_metric}_train.pkl', mode= 'wb') as file:
        pickle.dump(lgbm, file)

    # Checkpoint
    print(f'Models LGBM and RFE saved for {evaluation_metric}!')

# -----------------------------------------------------------------------------------