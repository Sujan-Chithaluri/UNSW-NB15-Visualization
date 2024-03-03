import os

import plotly
import plotly.graph_objs as go
import plotly.express as px

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from yellowbrick.cluster import InterclusterDistance

from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV

from umap import UMAP


class VizPlotting:
    
    def __init__(self):
        f1 = os.path.join('data', 'UNSW_NB15', 'UNSW_NB15_training-set.csv')
        f2 = os.path.join('data', 'UNSW_NB15', 'UNSW_NB15_features-set.csv')

        train_data = pd.read_csv(f1)
        feature_data = pd.read_csv(f2)

        self.name_description_map = dict(zip(feature_data['Name'], feature_data['Description']))

        actual_df = train_data

        train_data = train_data.drop(columns=['id'])
        train_data['attack_cat'] = train_data.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())
        train_data['is_ftp_login'] = np.where(train_data['is_ftp_login']>1, 1, train_data['is_ftp_login'])
        train_data['service'] = train_data['service'].apply(lambda x:"None" if x=="-" else x)

        for col in ['proto', 'service', 'state']:
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])

        cleaned_df = train_data
        reduced_df = pd.DataFrame()

        for attack_cat in train_data.attack_cat.unique():
            temp_df = train_data[train_data['attack_cat'] == attack_cat]
            temp_X = temp_df.drop('attack_cat', axis=1)
            temp_y = temp_df['attack_cat']
            temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(temp_X, temp_y, test_size=0.85, random_state=42)
            temp_reduced_df = temp_X_train
            temp_reduced_df['attack_cat'] = temp_y_train
            reduced_df = pd.concat([reduced_df, temp_reduced_df], ignore_index=True)
        

        reduced_X = reduced_df.drop('attack_cat', axis=1)
        reduced_y = reduced_df['attack_cat']
        corr_matrix = reduced_X.corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        reduced_X.drop(columns=to_drop, inplace=True)

        self.reduced_X = reduced_X
        self.reduced_y = reduced_y
        self.reduced_df = reduced_df
        self.actual_df = actual_df
        self.cleaned_df = cleaned_df

        self.create_kmeans()
        self.create_mse()
        self.create_apha()

        pass

    def create_intro_bargraph(self):

        train_data = self.cleaned_df
        temp_df = train_data['attack_cat'].value_counts().rename_axis('attack_cat').reset_index(name='counts')

        fig = px.bar(temp_df, x="attack_cat", y="counts", title="Attack class distribution")
        fig.update_layout(xaxis_title="Attack Categories", yaxis_title="Frequency")
        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON


    def create_pca_2d(self, real_time = False):

        filename = os.path.join('data', 'loaded', 'pca_2d.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty: 
            reduced_X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(reduced_X_scaled)

            temp_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df  

        fig = px.scatter(df, x='PC1', y='PC2', color='Target', title="PCA-2D (Principal Component Analysis 2D)")
        graphData = go.Figure(fig)
        graphJSON2D = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON2D


    def create_pca_3d(self, real_time = True):

        filename = os.path.join('data', 'loaded', 'pca_3d.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty: 
            reduced_X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(reduced_X_scaled)

            temp_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df

        fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Target',  title="PCA-3D (Principal Component Analysis 3D)")
        graphData = go.Figure(fig)
        graphJSON3D = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON3D


    def create_tsne(self, real_time = False):

        filename = os.path.join('data', 'loaded', 'tsne.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty:    
            reduced_X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            method = TSNE(n_components=2, random_state=42)
            components = method.fit_transform(reduced_X_scaled)
            
            temp_df = pd.DataFrame(components, columns=[f'Component{i+1}' for i in range(components.shape[1])])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df
        

        fig = px.scatter(df, x='Component1', y='Component2', color='Target',  title="TSNE (t-distributed Stochastic Neighbor Embedding)")
        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    
    def create_umap(self, real_time = False):

        filename = os.path.join('data', 'loaded', 'umap.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty: 
            reduced_X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            method = UMAP(n_components=2)
            components = method.fit_transform(reduced_X_scaled)
            
            temp_df = pd.DataFrame(components, columns=[f'Component{i+1}' for i in range(components.shape[1])])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df


        fig = px.scatter(df, x='Component1', y='Component2', color='Target',  title="UMAP (Uniform Manifold Approximation and Projection)")
        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON


    def create_lle(self, real_time = False):

        filename = os.path.join('data', 'loaded', 'lle.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty: 
            reduced_X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            method = LocallyLinearEmbedding(n_components=2, eigen_solver='dense', random_state=42)
            components = method.fit_transform(reduced_X_scaled)
            
            temp_df = pd.DataFrame(components, columns=[f'Component{i+1}' for i in range(components.shape[1])])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df

        fig = px.scatter(df, x='Component1', y='Component2', color='Target',   title="LLE (Locally Linear Embedding)")
        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON


    def create_svd(self, real_time = False):

        filename = os.path.join('data', 'loaded', 'svd.csv')
        df = pd.read_csv(filename) if os.path.exists(filename) else None

        if real_time or df is None or df.empty: 
            X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            method = TruncatedSVD(n_components=2)
            components = method.fit_transform(X_scaled)
            
            temp_df = pd.DataFrame(components, columns=[f'Component{i+1}' for i in range(components.shape[1])])
            temp_df['Target'] = self.reduced_y
            temp_df.to_csv(filename)
            
            df = temp_df

        fig = px.scatter(df, x='Component1', y='Component2', color='Target',   title="SVD (Singular Value Decomposition)")
        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON


    def create_numerical_plot(self, x_cat = "state", y_cat = "sbytes"):

        temp_df = self.reduced_df

        fig = px.box(temp_df, x=str(x_cat), y=str(y_cat), title="Box Plot")

        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON


    def create_categorical_plot(self, category="service"):

        col = str(category)
        description = self.name_description_map[col]

        temp_df = self.actual_df[col].value_counts().rename_axis(col).reset_index(name='counts')

        fig = px.bar(temp_df, x=col, y="counts", title=description.title() +" Distribution")
        fig.update_layout(xaxis_title=description, yaxis_title="Frequency")

        graphData = go.Figure(fig)
        graphJSON = json.dumps(graphData, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON
    
    def create_kmeans(self):
        
        filename = os.path.join('static', 'images', 'kmeans.png')

        if not os.path.exists(filename):

            X_scaled = MinMaxScaler().fit_transform(self.reduced_X)

            kmeans = KMeans(n_clusters=10, random_state=42)
            kmeans.fit_predict(X_scaled)

            visualizer = InterclusterDistance(kmeans)
            visualizer.fit(X_scaled)
            visualizer.show(outpath=filename)
    
    def create_mse(self):

        filename = os.path.join('static', 'images', 'mse.png')

        if not os.path.exists(filename):

            reduced_X = self.reduced_X
            reduced_y = self.reduced_y
            
            le = LabelEncoder()
            y = le.fit_transform(reduced_y)

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(reduced_X)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            fig = go.Figure()
            alphas = np.logspace(-4, 0, 100)
            enet_cv = ElasticNetCV(alphas=alphas, cv=5)
            enet_cv.fit(X_scaled, y)
            optimal_alpha = enet_cv.alpha_
            print("optimal_alpha",optimal_alpha)

            mse_scores = np.mean(enet_cv.mse_path_, axis=1)
            plt.figure(figsize=(12, 6))
            plt.plot(enet_cv.alphas_, enet_cv.mse_path_, ':')
            plt.plot(enet_cv.alphas_, mse_scores, 'k', label='Average across the folds', linewidth=2)
            plt.axvline(optimal_alpha, linestyle='--', color='k', label='Optimal alpha')
            plt.xlabel('-log(alpha)')
            plt.ylabel('Mean squared error')
            plt.legend()
            plt.savefig(filename)
            plt.close()
    
    def create_apha(self):

        filename = os.path.join('static', 'images', 'alpha.png')

        if not os.path.exists(filename):
            
            X_scaled = MinMaxScaler().fit_transform(self.reduced_X)
            reduced_y = self.reduced_y
            
            le = LabelEncoder()
            y = le.fit_transform(reduced_y)

            alphas = np.logspace(-4, 0, 100)
            ceofs_list = []
            for alpha in alphas:
                enet_cv = ElasticNetCV(alphas=[alpha], l1_ratio=0.5, cv=5)
                enet_cv.fit(X_scaled, y)
                coefs = enet_cv.coef_
                ceofs_list.append(coefs)
            numpy_array = np.array(ceofs_list)
            transposed_array = numpy_array.T
            coefficients_list = transposed_array.tolist()
            fig_paths, ax = plt.subplots(figsize=(10, 6))

            for i in range(coefs.shape[0]):
                ax.plot(-np.log(alphas), coefficients_list[i])

            ax.set_title('Regularization Paths - Coefficients of Correlated Features')
            ax.set_xlabel('-log10(alpha)')
            ax.set_ylabel('Coefficient Value')
            ax.legend()
            plt.savefig(filename)
            plt.close()


