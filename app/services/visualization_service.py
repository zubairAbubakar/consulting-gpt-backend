import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from app.models.technology import MarketAnalysis, PCAResult

class VisualizationService:
    def prepare_axes_plot_data(self, market_analyses: List[MarketAnalysis]) -> Dict:
        """
        Prepare data for raw axes visualization
        """
        data_dict = {
            "technologies": [],
            "axes": [],
            "scores": []
        }
        
        for analysis in market_analyses:
            data_dict["technologies"].append(analysis.related_technology.name)
            data_dict["axes"].append(analysis.comparison_axis.axis_name)
            data_dict["scores"].append(analysis.score)
            
        return data_dict

    def prepare_pca_plot_data(self, pca_result: PCAResult) -> Dict:
        """
        Prepare data for PCA visualization
        """
        return {
            "transformed_data": pca_result.transformed_data,
            "variance_explained": [
                comp["explained_variance_ratio"] 
                for comp in pca_result.components
            ],
            "loadings": {
                f"PC{i+1}": comp["loadings"]
                for i, comp in enumerate(pca_result.components)
            }
        }
    
    def prepare_pca_plot_data_2d(self, pca_result: PCAResult) -> Dict:
        """
        Prepare PCA visualization data
        """
        # 1. Technology positions in PC space
        scatter_data = {
            "points": [],
            "labels": []
        }
        
        for tech_name, coords in pca_result.transformed_data.items():
            scatter_data["points"].append(coords)
            scatter_data["labels"].append(tech_name)
            
        # 2. Loading vectors for original axes
        loading_vectors = []
        for comp in pca_result.components:
            if "loadings" in comp:
                for axis_name, loading in comp["loadings"].items():
                    loading_vectors.append({
                        "axis": axis_name,
                        "pc1_loading": loading["PC1"],
                        "pc2_loading": loading["PC2"]
                    })
        
        # 3. Variance explained
        variance_explained = [
            comp["explained_variance_ratio"]
            for comp in pca_result.components
        ]
        
        return {
            "scatter_data": scatter_data,
            "loading_vectors": loading_vectors,
            "variance_explained": variance_explained
        }
    