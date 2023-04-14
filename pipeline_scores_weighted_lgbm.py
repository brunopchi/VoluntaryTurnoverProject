# Básico
import pickle
import json
import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt

# Visualização
import matplotlib
import matplotlib.pyplot as plt

# Pipelines, Transformadores e Modelos
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# Métricas de avaliação
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix
)

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


def easy_ordinal_encoder(df_x_train, df_x_test, ordinal_columns, ordinal_categories):
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
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
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
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (testing set).
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

    # Apply the transformation in train and test data
    encoded_array_train = encoder.transform(df_x_train.loc[:,ordinal_columns]) 
    encoded_array_test = encoder.transform(df_x_test.loc[:,ordinal_columns])

    # Build the dataframes preserving original indexes
    df_x_train[ordinal_columns] = pd.DataFrame(encoded_array_train,columns=encoder.get_feature_names_out(), index=df_x_train.index)
    df_x_test[ordinal_columns] = pd.DataFrame(encoded_array_test,columns=encoder.get_feature_names_out(), index=df_x_test.index)
    
    with open('./encoders_scalers/ordinal_encoder.pkl', mode='wb') as file:
        pickle.dump(encoder, file)
    
    return df_x_train, df_x_test


def easy_leave_one_out_encoder(df_x_train, df_x_test, df_y_train, method='auto', nominal_columns=[]):
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
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
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
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns encoded (testing set).
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

    # Apply the transformation in train and test data
    df_x_train = encoder.transform(df_x_train)
    df_x_test = encoder.transform(df_x_test)
    
    with open('./encoders_scalers/leave_one_out_encoder.pkl', mode='wb') as file:
        pickle.dump(encoder, file)
    
    return df_x_train, df_x_test


def easy_standard_scaler(df_x_train, df_x_test, method='auto', numerical_columns=[]):
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
    df_x_test : pandas.core.frame.DataFrame
        Pandas dataframe containing all independent variables for testing.
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
    df_x_test: pandas.core.frame.DataFrame
        Pandas dataframe with the given columns scaled (testing set).
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
        
        # Apply the scaling in train and test data
        df_x_train[continuous_variable] = scaler.transform(df_x_train[continuous_variable].array.reshape(-1,1))
        df_x_test[continuous_variable] = scaler.transform(df_x_test[continuous_variable].array.reshape(-1,1))
        
        with open(f'./encoders_scalers/standard_scaler_{continuous_variable}.pkl', mode='wb') as file:
            pickle.dump(scaler, file)
        
    return df_x_train, df_x_test


def opt_moving_thresh_roc(y_test, y_pred):
    """
    Computes the optimal probabilistc threshold base on the ROC curve.
    For more information check the links below:
    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    
    Necessary Packages
    ------------------
    Numpy:
        import numpy as np
    Sklearn:
        from sklearn.metrics import roc_curve
    
    Parameters
    ---------
    y_test : pandas.core.series.Series
        Pandas series containing the dependent variable for test. 
    y_pred : numpy.ndarray
        Array with predictions values inferred on testing data by 
        a trained model.
        
    Returns
    -------
    threshold_opt_roc: numpy.float64
        Float number that represents the optimal probabilistic 
        threshold based on the ROC curve.
    """
        
    # Construction of the ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # Computes the geometric mean
    gmean = np.sqrt(tpr * (1 - fpr))

    # Search for the optimal threshold
    index = np.argmax(gmean)
    threshold_opt_roc = round(thresholds[index], ndigits = 4)
    
    return threshold_opt_roc


def opt_moving_thresh_pr(y_test, y_pred):
    """
    Computes the optimal probabilistc threshold base on the Precision-Recall curve.
    For more information check the links below:
    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    
    Necessary Packages
    ------------------
    Numpy:
        import numpy as np
    Sklearn:
        from sklearn.metrics import precision_recall_curve
    
    Parameters
    ---------
    y_test : pandas.core.series.Series
        Pandas series containing the dependent variable for test. 
    y_pred : numpy.ndarray
        Array with predictions values inferred on testing data by 
        a trained model.
        
    Returns
    -------
    threshold_opt_pr: numpy.float64
        Float number that represents the optimal probabilistic 
        threshold based on the Precision-Recall curve.
    """
    
    # Construction of the Precision-Recall curve data
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

    # Computes the f-score
    fscore = (2 * precision * recall) / (precision + recall)

    # Search for the optimal threshold
    index = np.argmax(fscore)
    threshold_opt_pr = round(thresholds[index], ndigits = 4)
    
    return threshold_opt_pr


def opt_moving_thresh_fs(y_test, y_pred):
    """
    Computes the optimal probabilistc threshold base on the F-Score curve.
    For more information check the links below:
    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    
    Necessary Packages
    ------------------
    Numpy:
        import numpy as np
    Sklearn:
        from sklearn.metrics import f1_score
    
    Parameters
    ---------
    y_test : pandas.core.series.Series
        Pandas series containing the dependent variable for test. 
    y_pred : numpy.ndarray
        Array with predictions values inferred on testing data by 
        a trained model.
        
    Returns
    -------
    threshold_opt_fs: numpy.float64
        Float number that represents the optimal probabilistic 
        threshold based on the F-Score curve.
    """    
    
    # Array for finding the optimal threshold
    thresholds = np.arange(0.0, 1.0, 0.0001)
    fscore = np.zeros(shape=(len(thresholds)))
    
    # Fit the model
    for index, elem in enumerate(thresholds):
        
        # Corrected probabilities
        y_pred_prob = (y_pred > elem).astype('int')
        
        # Calculate the f-score
        fscore[index] = f1_score(y_test, y_pred_prob)

    # Search for the optimal threshold
    index = np.argmax(fscore)
    threshold_opt_fs = round(thresholds[index], ndigits = 4)

    return threshold_opt_fs


def shap_summary_plot(model, df_x_train, df_x_test, model_name):
    """
    Save a feature importance plot based on shapley values in './modelo_explicabilidade/'. 
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
    model_name : str
        Name of the saved model.
        
    Returns
    -------
    None
    """
    
    # Fit TreeExplainer
    explainer = shap.TreeExplainer(model, df_x_train, feature_names=df_x_train.columns.tolist(), model_output='probability')
    
    # Compute shapley values
    shap_values = explainer.shap_values(df_x_test)
    
    # Build a the feature importance
    shap.summary_plot(shap_values, df_x_test, feature_names=df_x_test.columns, plot_type="bar", show=False)
    
    # Save a pdf for future reference
    plt.savefig(f'./model_metrics/summary_plot_{model_name}_weighted.pdf', format='pdf', dpi=600, bbox_inches='tight')
    
    return None

def model_metrics_func(true_class, predicted_class):
    """
    Computes the most relevant metrics for classification tasks.

    Necessary Packages
    ------------------
    Sklearn:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
            precision_recall_curve,
            f1_score,
            classification_report,
            confusion_matrix
        )

    Parameters
    ----------
    true_class : pandas.core.series.Series
        Object which contains the true classes of the dependent variable.
    predicted_class : numpy.ndarray
        Object which contains the predicted classes of the dependent variable.

    Returns
    -------
    model_metrics_dict : dict
        Dicitionary containing the most relevant classification metrics.
    """

    model_accuracy_score = accuracy_score(true_class, predicted_class)
    model_precision_score = precision_score(true_class, predicted_class)
    model_recall_score = recall_score(true_class, predicted_class)
    model_roc_auc_score = roc_auc_score(true_class, predicted_class)
    model_f1_score = f1_score(true_class, predicted_class)
    model_error = 1 - model_accuracy_score
    model_confidence_interval = 1.96*sqrt((model_error*(1-model_error))/true_class.shape[0]) # (95% confidence)
    model_confusion_matrix = confusion_matrix(true_class, predicted_class)
    model_classification_report = classification_report(true_class, predicted_class)


    model_metrics_dict = {
        'model_accuracy_score' : model_accuracy_score,
        'model_precision_score' : model_precision_score,
        'model_recall_score' : model_recall_score,
        'model_roc_auc_score' : model_roc_auc_score,
        'model_f1_score' : model_f1_score,
        'model_error' : model_error,
        'model_confidence_interval' : model_confidence_interval,
        'model_confusion_matrix' : model_confusion_matrix,
        'model_classification_report' : model_classification_report
    }

    return model_metrics_dict

# ------------------------- Classification metrics for GridSearchCV -------------------------

# Classifier evaluation metric
evaluation_metric_list = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'neg_log_loss']

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
    X_df, y_df = easy_target_variable_organizer(df=df, target_variable='SaidaVoluntaria')

    # Stratified split of the database between training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.30, stratify = y_df, random_state=42)

    # Save raw data to GridSearchCV
    with open('./database/dados_gridsearchcv.pkl', mode= 'wb') as file:
        pickle.dump([X_train, y_train, X_test, y_test], file)

    # Checkpoint
    print('Raw data saved!')

# --------- Pipeline with output for model training and testing after hyperparameter calibration ---------

    # Missing data handling
    # Train
    X_train['nom_grupo_cargos'] = X_train['nom_grupo_cargos'].fillna('NAO INFORMADO')
    X_train['Custo_Benef_Mes_Anterior'] = X_train['Custo_Benef_Mes_Anterior'].fillna(round(X_train['Custo_Benef_Mes_Anterior'].median(), 2))
    X_train['Num_Benef_Mes_Anterior'] = X_train['Num_Benef_Mes_Anterior'].fillna(round(X_train['Num_Benef_Mes_Anterior'].median(), 2))

    # Test
    X_test['nom_grupo_cargos'] = X_test['nom_grupo_cargos'].fillna('NAO INFORMADO')
    X_test['Custo_Benef_Mes_Anterior'] = X_test['Custo_Benef_Mes_Anterior'].fillna(round(X_test['Custo_Benef_Mes_Anterior'].median(), 2))
    X_test['Num_Benef_Mes_Anterior'] = X_test['Num_Benef_Mes_Anterior'].fillna(round(X_test['Num_Benef_Mes_Anterior'].median(), 2))


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

    X_train, X_test = easy_ordinal_encoder(
        df_x_train=X_train,
        df_x_test=X_test,
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
        'nom_ger',
        'dsc_estrutura_n1',
        'dsc_estrutura_n2'
    ]

    X_train, X_test = easy_leave_one_out_encoder(
        df_x_train=X_train,
        df_x_test=X_test,
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

    X_train, X_test = easy_standard_scaler(
        df_x_train=X_train,
        df_x_test=X_test,
        method='declarative',
        numerical_columns=columns_numerical
    )

    # Saves train and test data
    with open('./database/dados_pre_processados.pkl', mode= 'wb') as file:
        pickle.dump([X_train, y_train, X_test, y_test], file)

    # Checkpoint
    print('Pre-processed data saved!')

# -------------- Processing pipeline (gridsearch, training, testing and evaluation) -------------

    # Input for hyperparameter optimization
    with open('./database/dados_gridsearchcv.pkl', mode= 'rb') as file:
        X_train, y_train, X_test, y_test = pickle.load(file)    

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
        'nom_ger',
        'dsc_estrutura_n1',
        'dsc_estrutura_n2'
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
    with open('./database/dados_pre_processados.pkl', mode= 'rb') as file:
        X_train, y_train, X_test, y_test = pickle.load(file)

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
    with open(f'./trained_models/rfe_weighted_{evaluation_metric}.pkl', mode= 'wb') as file:
        pickle.dump(rfe, file)

    # Selection of the most relevant features
    X_train = X_train.loc[:, rfe.support_]
    X_test = X_test.loc[:, rfe.support_]

    # Train the model
    lgbm.fit(X_train, y_train)

    # Save trained model
    with open(f'./trained_models/lgbmclassifier_weighted_{evaluation_metric}.pkl', mode= 'wb') as file:
        pickle.dump(lgbm, file)

    # Checkpoint
    print(f'Models LGBM and RFE saved for {evaluation_metric}!')

    # Saves relevance of features for reference
    shap_summary_plot(lgbm, X_train, X_test, f'{model_name}_{evaluation_metric}')

    # Class prediction of the minority class (1) for train set
    y_pred_train = lgbm.predict(X_train)

    # Probability prediction of the minority class (1) for test set
    y_pred_test = lgbm.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------------------------------------

    # ------------------- Optimization of the probabilistic threshold -------------------

    opt_thresholds = {
        'noopt' : 0.5,
        'optroc' : opt_moving_thresh_roc(y_test, y_pred_test),
        'optprecisionrecall' : opt_moving_thresh_pr(y_test, y_pred_test),
        'optfscore' : opt_moving_thresh_fs(y_test, y_pred_test)
    }

    # Checkpoint
    print(f'Optimal probabilistic thresholds computed for {evaluation_metric}!')

    # -----------------------------------------------------------------------------------

    # ---------------- Output: model and probabilistic adjustment method ----------------

    for thresh_name, thresh_value in opt_thresholds.items():

        # Results with adjustment of the probabilistic threshold
        y_pred_test_tuned_thresh = np.where(y_pred_test >= thresh_value, 1, 0)

        # Computes the most relevant metrics of a classifier based on train set
        model_metrics_dict_train = model_metrics_func(y_train, y_pred_train)
        
        # Computes the most relevant metrics of a classifier based on test set
        model_metrics_dict_test = model_metrics_func(y_test, y_pred_test_tuned_thresh)

        streamlit_metrics_dict = {
            'model' : model_name,
            'grid_evaluation_metric' : evaluation_metric,
            'optimization': thresh_name,
            'prob_treshold': thresh_value,
            'accuracy': model_metrics_dict_test.get('model_accuracy_score'),
            'precision': model_metrics_dict_test.get('model_precision_score'),
            'recall' : model_metrics_dict_test.get('model_recall_score'),
            'rocauc': model_metrics_dict_test.get('model_roc_auc_score'),
            'f1': model_metrics_dict_test.get('model_f1_score'),
            'error': model_metrics_dict_test.get('model_error'),
            'conf_interval': model_metrics_dict_test.get('model_confidence_interval')
        }

        # Convert dictionary to pandas dataframe
        df_streamlit_metrics = pd.DataFrame([streamlit_metrics_dict], columns = streamlit_metrics_dict.keys())

        # Saves model scores in .pkl format
        with open(f'./model_metrics/{model_name}_{evaluation_metric}_{thresh_name}_scores.pkl', mode= 'wb') as file:
            pickle.dump(df_streamlit_metrics, file)

        # Saves model scores in .txt format
        with open(f'./model_metrics/resultados_{model_name}_{evaluation_metric}_{thresh_name}.txt', mode='w') as report:
            print(f'Model: {model_name}', file=report)
            print(f'Model best parameters {grid_search.best_params_} \n', file=report)
            print(f'Database used: Test', file=report)
            print(f'GridSearchCV evaluation metric: {evaluation_metric}', file=report)
            print(f'Otimizador: {thresh_name} \n', file=report)
            print('Confusion Matrix:', file=report)
            print(model_metrics_dict_test.get('model_confusion_matrix'), file=report)
            print('\n Classification report:', file=report)
            print(model_metrics_dict_test.get('model_classification_report'), file=report)
            print('Other metrics: \n', file=report)
            print(f'Optimal probabilistic treshold: {thresh_value}', file=report)
            print('Overall Accuracy score for test set: {0:.4f}'.format(model_metrics_dict_test.get('model_accuracy_score')), file=report)
            print('Overall ROCAUC score for test set: {0:.4f}'.format(model_metrics_dict_test.get('model_roc_auc_score')), file=report)
            print('Overall F1 score for test set: {0:.4f}'.format(model_metrics_dict_test.get('model_f1_score')), file=report)
            print(f"Error:{model_metrics_dict_test.get('model_error')}", file=report)
            print(f"Confidence interval:{model_metrics_dict_test.get('model_confidence_interval')} \n", file=report)
            print('------------------------------------------------------------ \n', file=report)
            print(f'Model: {model_name}', file=report)
            print(f'Database used: Train', file=report)
            print(f'GridSearchCV evaluation metric: {evaluation_metric}', file=report)
            print(f'Otimizador: {thresh_name} \n', file=report)
            print('Confusion Matrix:', file=report)
            print(model_metrics_dict_train.get('model_confusion_matrix'), file=report)
            print('\n Classification report:', file=report)
            print(model_metrics_dict_train.get('model_classification_report'), file=report)
            print('Other metrics: \n', file=report)
            print(f'Optimal probabilistic treshold: {thresh_value}', file=report)
            print('Overall Accuracy score for test set: {0:.4f}'.format(model_metrics_dict_train.get('model_accuracy_score')), file=report)
            print('Overall ROCAUC score for test set: {0:.4f}'.format(model_metrics_dict_train.get('model_roc_auc_score')), file=report)
            print('Overall F1 score for test set: {0:.4f}'.format(model_metrics_dict_train.get('model_f1_score')), file=report)
            print(f"Error:{model_metrics_dict_train.get('model_error')}", file=report)
            print(f"Confidence interval:{model_metrics_dict_train.get('model_confidence_interval')} \n", file=report)
            print('------------------------------------------------------------', file=report)

    # Checkpoint
    print('Models metrics saved!')
    print(f'Metric {evaluation_metric} done!')

# -----------------------------------------------------------------------------------