# Básico
import pickle
import pandas as pd
import numpy as np

# Transformers
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import LeaveOneOutEncoder

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

# -------------------------- Pipeline with output for GridSearchCV --------------------------

# Import abt
X_prod = pd.read_csv('./database/abt_prod_data.csv')

# Save raw data to GridSearchCV
with open('./database/dados_raw_prod.pkl', mode= 'wb') as file:
    pickle.dump(X_prod, file)

# Checkpoint
print('Raw data saved!')

# Missing data handling
X_prod['nom_grupo_cargos'] = X_prod['nom_grupo_cargos'].fillna('NAO INFORMADO')
X_prod['Custo_Benef_Mes_Anterior'] = X_prod['Custo_Benef_Mes_Anterior'].fillna(round(X_prod['Custo_Benef_Mes_Anterior'].median(), 2))
X_prod['Num_Benef_Mes_Anterior'] = X_prod['Num_Benef_Mes_Anterior'].fillna(round(X_prod['Num_Benef_Mes_Anterior'].median(), 2))

# --------- Pipeline with output for model training and testing after hyperparameter calibration ---------

ordinal_columns = [
    'nom_grupo_cargos',
    'nom_grau_instrucao',
    'num_nivel_hierarquico',
    'nom_posicao'
]

# Apply the transformation to the ordinal independent variables
X_prod = apply_ordinal_encoder(ordinal_encoder, X_prod, ordinal_columns)

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
X_prod = apply_nominal_encoder(leave_one_out_encoder, X_prod)

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
X_prod = apply_scalers(X_prod, continuous_scalers)

# Saves train data
with open('./database/dados_preprocess_prod.pkl', mode= 'wb') as file:
    pickle.dump(X_prod, file)

# Checkpoint
print('Pre-processed data saved!')