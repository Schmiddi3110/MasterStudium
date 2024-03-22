import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def combine_data(path, filestart):
    """
    Add all csv files in the folder into one dataframe (if it starts with "filestart"). 
    Parameters:
        path: Path to the csv files
        filestart: Start of the filenames that should be added (B1 for first Bundesliga)
    Returns:
        combined_data: All contents of the csv files in one dataframe
    """
    #Alle Saisonspieldateinamen in eine Liste laden
    season_games_files = [file for file in os.listdir(path) if file.endswith('.csv') and file.startswith(filestart)]
    
    combined_data = pd.DataFrame()
    
    #Alle Dateien zu einem großen Dataframe zusammenfügen
    for file in season_games_files:
        file_path = os.path.join(path, file)
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    #Dateien zurückgeben
    return combined_data

def plot_distributions(feature_df, features_desc, figname, max_cols_row=3):
    """
    Plot all distributions of the features in feature_df.
    Parameters:
        feature_df: DataFrame with all features distributions should be created for
        features_desc: Title for each plot
        figname: Name of the png-file of the figure
        max_cols_row: Max amount of plots that can be shown next to each other
    """
    figpath = 'plots/Verteilungen/'
    long_feature_names = ['HomeTeam', 'AwayTeam']
    sb.set_style("whitegrid")
    
    #Cont rows/cols
    num_cols = len(feature_df.columns)
    num_rows = -(-num_cols // max_cols_row)
    
    #Init grid
    fig, axes = plt.subplots(num_rows, max_cols_row, figsize=(20, 4 * num_rows))
    axes = axes.flatten() 
    
    #Plot a Histogram for each feature
    for i, column in enumerate(feature_df.columns):
        sb.histplot(feature_df[column], kde=True, bins=10, color='skyblue', ax=axes[i])
        axes[i].set_title(features_desc[i])
        
        if column in long_feature_names:
            axes[i].tick_params(axis='x', rotation=90)
        elif column == 'Date':
            axes[i].set_xticklabels([])
        else:
            axes[i].set_xlabel(column)
            
        axes[i].set_ylabel('Häufigkeit')
    
    #Rem empty subplots 
    for i in range(num_cols, num_rows * max_cols_row):
        fig.delaxes(axes[i])

    #Save the plots
    fig.savefig(f'{figpath}{figname}.png')

    plt.tight_layout()
    plt.show()

def plot_heatmap(feature_df, title, save_path, figsize=(10, 8), fontsize=12, annot_kws={"size": 16}, size=12, font_scale=1.5):
    """
    Plot a heatmap for all features in feature_df
    Parameter:
        feature_df: DataFrame with all features that should be included in the heatmap
        title: Title of the heatmap
        save_path: Path where to save the figure png file to
        figsize: Size of the heatmap
        fontsize: Fontsize for the title
        annot_kws: Fontsize for labels of the heatmap
        size: Fontsize for labels
        font_scale: Scale of the font for the legend
    """
    corr_matrix = feature_df.corr()

    plt.figure(figsize=figsize)
    sb.set(font_scale=font_scale)
    
    heatmap = sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws=annot_kws)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), size=size)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), size=size)

    plt.title(title, fontsize=fontsize)
    plt.savefig(save_path)
    plt.show()

def plot_pairplot(feature_df, save_path, fontsize=12, font_scale=1.5):
    """
    Plot a pairplot for all features in feature_df
    Parameter:
        feature_df: DataFrame with all features that should be included in the pairplot
        save_path: Path where to save the figure png file to
        fontsize: Fontsize for the title
        font_scale: Scale of the font for the legend
    """
    sb.pairplot(feature_df)
    sb.set(font_scale=font_scale)
    plt.savefig(save_path)
    plt.show()

def construct_preprocessor(numeric_features, categorical_features):
    """
    Returns a preprocessor for numeric and categorical features.
    Parameters:
        numeric_features: Numeric features of the Dataframe
        categorical_features: Categorical features of the Dataframe
    """
    #Pipeline for numerical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
        
    #Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
        
    #Apply pipelines to their columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def plot_conf_matrices(conf_matr_dt, conf_matr_svm, conf_matr_rf, save_path):
    """
    Plot the provided confusion matrices for the three models.
    Parameters:
        conf_matr_dt: Confusion matrice for the DecisionTree
        conf_matr_svm: Confusion matrice for the SVM
        conf_matr_rf: Confusion matrice for the RandomForest
    """
    #Configure grid
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    #Conf-matrix for dt
    sb.heatmap(conf_matr_dt, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['A(0)', 'D(1)', 'H(2)'],
                yticklabels=['A(0)', 'D(1)', 'H(2)'], ax=axes[0])
    axes[0].set_title('Decision Tree')
    
    #Conf-matrix for svm
    sb.heatmap(conf_matr_svm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['A(0)', 'D(1)', 'H(2)'],
                yticklabels=['A(0)', 'D(1)', 'H(2)'], ax=axes[1])
    axes[1].set_title('SVM')
    
    #Conf-matrix for rf
    sb.heatmap(conf_matr_rf, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['A(0)', 'D(1)', 'H(2)'],
                yticklabels=['A(0)', 'D(1)', 'H(2)'], ax=axes[2])
    axes[2].set_title('Random Forest')
    plt.savefig(save_path)
    plt.show()
    