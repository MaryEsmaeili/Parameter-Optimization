import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    A class for data processing and feature engineering tailored to the lung dataset.
    
    This class handles data preprocessing steps including feature selection, encoding, log transformation,
    scaling, and dimensionality reduction (PCA) as specified in the assignment criteria.
    """

    def __init__(self, gene_expression, lung_metadata):
        """
        Initializes the DataProcessor with gene expression and metadata DataFrames.
        
        Parameters:
            gene_expression (pd.DataFrame): The gene expression data.
            lung_metadata (pd.DataFrame): The metadata associated with the gene expression data.
        """
        self.gene_expression = gene_expression
        self.lung_metadata = lung_metadata.copy()
    
    def run_pca(self, variance_threshold=0.95):
        """
        Runs PCA on the gene expression data and returns the PCA DataFrame and model.

        Parameters:
            variance_threshold (float): The amount of variance to retain during PCA.

        Returns:
            pd.DataFrame: DataFrame containing PCA-transformed features.
        """
        transposed_data = self.gene_expression.iloc[1:, 1:].T.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
        pca = PCA(n_components=variance_threshold)
        pca_result = pca.fit_transform(transposed_data)
        pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
        return pca_df, pca
    
    def drop_single_unique_value_columns(self):
        """
        Drops columns from lung_metadata with a single unique value.
        """
        cols_to_drop = [col for col in self.lung_metadata.columns if self.lung_metadata[col].nunique() <= 1]
        self.lung_metadata.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    def encode_features(self):
        """
        Encodes categorical features in lung_metadata.
        """
        gender_mapping = {'M': 1, 'F': 0}
        stage_mapping = {'pT1': 1, 'pT2': 2, 'pT3': 3, 'pTX': 0, 'pN0': 0, 'pN1': 1, 'pNX': 0, 'pM0': 0, 'pM1': 1, 'pMX': 0}
        
        self.lung_metadata['source.location'] = self.lung_metadata['source.location'].astype('category').cat.codes
        self.lung_metadata['characteristics.tag.gender'] = self.lung_metadata['characteristics.tag.gender'].map(gender_mapping)
        
        unique_histology_values = self.lung_metadata['characteristics.tag.histology'].unique()
        histology_mapping = {value: idx for idx, value in enumerate(unique_histology_values, start=1)}
        histology_mapping['Not Available'] = 0
        self.lung_metadata['characteristics.tag.histology'] = self.lung_metadata['characteristics.tag.histology'].map(histology_mapping).fillna(0)

        self.lung_metadata['characteristics.tag.stage.primary.tumor'] = self.lung_metadata['characteristics.tag.stage.primary.tumor'].map(stage_mapping).fillna(0)
        self.lung_metadata['characteristics.tag.stage.nodes'] = self.lung_metadata['characteristics.tag.stage.nodes'].map(stage_mapping).fillna(0)
        self.lung_metadata['characteristics.tag.stage.mets'] = self.lung_metadata['characteristics.tag.stage.mets'].map(stage_mapping).fillna(0)
        
        self.lung_metadata['characteristics.tag.grade'] = self.lung_metadata['characteristics.tag.grade'].replace("Not Available", 0)
    
    def create_binary_features(self):
        """
        Creates binary features in lung_metadata.
        """
        if 'characteristics.tag.tumor.size.maximumdiameter' in self.lung_metadata.columns:
            self.lung_metadata['characteristics.tag.tumor.size.maximumdiameter'] = np.where(
                self.lung_metadata['characteristics.tag.tumor.size.maximumdiameter'] > 5, 1, 0
            )
    
    def merge_dataframes(self, pca_df):
        """
        Merges the PCA-transformed gene expression data with lung metadata.
        
        Parameters:
            pca_df (pd.DataFrame): DataFrame of PCA-transformed features.

        Returns:
            pd.DataFrame: Merged DataFrame with both PCA and lung metadata features.
        """
        self.lung_metadata.drop(columns=['title', 'CEL.file'], inplace=True, errors='ignore')
        combined_df = pd.merge(self.lung_metadata, pca_df, left_index=True, right_index=True, how='inner')
        return combined_df.dropna()
    
    def scale_and_transform(self, combined_df):
        """
        Applies log transformation and scaling to specified columns in the combined DataFrame.

        Parameters:
            combined_df (pd.DataFrame): The combined DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        if (combined_df['characteristics.tag.tumor.size.maximumdiameter'] > 0).all():
            combined_df['log_tumor_size'] = np.log(combined_df['characteristics.tag.tumor.size.maximumdiameter'])

        scaler = StandardScaler()
        
        # Dynamically select available PCA components for scaling
        pca_columns = [col for col in combined_df.columns if col.startswith('PC')]
        columns_to_scale = ['characteristics.tag.tumor.size.maximumdiameter'] + pca_columns
        combined_df[columns_to_scale] = scaler.fit_transform(combined_df[columns_to_scale])
        
        return combined_df

