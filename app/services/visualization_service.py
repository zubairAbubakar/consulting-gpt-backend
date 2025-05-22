import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.models.technology import ClusterResult, MarketAnalysis, PCAResult
from app.schemas.visualization import TechnologyDetail

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationService:
    def prepare_visualization_data(
        self,
        pca_result: Optional[PCAResult] = None,
        market_analyses: Optional[List[MarketAnalysis]] = None,
        cluster_results: Optional[List[ClusterResult]] = None,
        show_clusters: bool = False,
        selected_axes: Optional[List[str]] = None,
        show_labels: bool = True,
        show_annotations: bool = True
    ) -> Dict:
        """Enhanced visualization data preparation"""
        
        result = {
            "type": "combined" if pca_result and market_analyses else "pca" if pca_result else "raw",
            "interactive": {
                "show_labels": show_labels,
                "show_annotations": show_annotations,
                "selected_axes": selected_axes or [],
                "click_details": {}
            }
        }

        # Add PCA view if available
        if pca_result:
            pca_data = self.prepare_pca_plot_data_2d(pca_result)
            result["pca_view"] = pca_data

        # Add raw axes view if available
        if market_analyses:
            raw_data = self.prepare_axes_plot_data(
                market_analyses, 
                selected_axes=selected_axes
            )
            result["raw_view"] = raw_data

        # Add cluster information if requested
        if show_clusters and cluster_results:
            cluster_data = self._prepare_cluster_visualization(
                cluster_results,
                pca_result.transformed_data if pca_result else None
            )
            result["cluster_view"] = cluster_data

        # Add click details for each technology
        result["interactive"]["click_details"] = self._prepare_technology_details(
            market_analyses, 
            cluster_results
        )

        return result

    def _prepare_cluster_visualization(
        self,
        cluster_results: List[ClusterResult],
        transformed_data: Optional[Dict] = None
    ) -> Dict:
        """Prepare cluster visualization data"""
        return {
            "clusters": [
                {
                    "id": cluster.id,
                    "name": cluster.name,
                    "center": [cluster.center_x, cluster.center_y],
                    "spread": cluster.cluster_spread,
                    "contains_target": cluster.contains_target,
                    "members": [
                        {
                            "name": member.related_technology.name,
                            "distance": member.distance_to_center,
                            "coords": transformed_data.get(member.related_technology.name, [])
                            if transformed_data else []
                        }
                        for member in cluster.cluster_members
                    ]
                }
                for cluster in cluster_results
            ]
        }

    def _prepare_technology_details(
        self,
        market_analyses: Optional[List[MarketAnalysis]],
        cluster_results: Optional[List[ClusterResult]]
    ) -> Dict[str, TechnologyDetail]:
        """Prepare technology details for interactive features"""
        details = {}
        
        if market_analyses:
            for analysis in market_analyses:
                tech = analysis.related_technology
                if tech.name not in details:
                    details[tech.name] = {
                        "name": tech.name,
                        "abstract": tech.abstract,
                        "market_scores": {}
                    }
                details[tech.name]["market_scores"][analysis.comparison_axis.axis_name] = analysis.score

        if cluster_results:
            for cluster in cluster_results:
                for member in cluster.cluster_members:
                    tech = member.related_technology
                    if tech.name in details:
                        details[tech.name]["cluster_name"] = cluster.name

        return details

    def _prepare_clustered_view(
        self,
        pca_result: PCAResult,
        cluster_results: List[ClusterResult]
    ) -> Dict:
        """
        Prepare visualization data with cluster information
        """
        return {
            "type": "clustered",
            "points": pca_result.transformed_data,
            "clusters": [
                {
                    "id": cluster.id,
                    "name": cluster.name,
                    "center": [cluster.center_x, cluster.center_y],
                    "members": [
                        {
                            "name": member.related_technology.name,
                            "coords": pca_result.transformed_data[member.related_technology.name]
                        }
                        for member in cluster.cluster_members
                    ]
                }
                for cluster in cluster_results
            ],
            "variance_explained": [
                comp["explained_variance_ratio"]
                for comp in pca_result.components
            ]
        }
    
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
        Prepare 2D PCA plot data
        """
        # Format loadings for each component
        formatted_loadings = []
        for component in pca_result.components[:2]:  # Only first 2 components
            for axis_name, loading_value in component["loadings"].items():
                formatted_loadings.append({
                    "axis": axis_name,
                    "pc1_loading": loading_value if component["component_number"] == 1 else 0,
                    "pc2_loading": loading_value if component["component_number"] == 2 else 0
                })

        return {
            "type": "pca",
            "points": pca_result.transformed_data,
            "loadings": formatted_loadings,
            "variance_explained": [
                comp["explained_variance_ratio"]
                for comp in pca_result.components[:2]
            ]
        }
    
    def prepare_silhouette_analysis(
        self,
        pca_result: PCAResult,
        max_clusters: int = 10
    ) -> Dict:
        """
        Prepare silhouette analysis data for clustering
        """
        points = np.array([coords for coords in pca_result.transformed_data.values()])
        n_samples = points.shape[0]
        scores = []
        
        # Adjust max_clusters to be valid for silhouette analysis
        # Valid values are 2 to n_samples - 1 (inclusive)
        max_possible_clusters = min(max_clusters, n_samples - 1)
        
        if max_possible_clusters < 2:
            return {
                "type": "silhouette_analysis",
                "scores": [],
                "optimal_k": 2,
                "error": "Not enough samples for silhouette analysis"
            }

        for n_clusters in range(2, max_possible_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(points)
                silhouette_avg = silhouette_score(points, cluster_labels)
                scores.append({
                    "n_clusters": n_clusters,
                    "score": float(silhouette_avg)
                })
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for k={n_clusters}: {e}")
                continue
        
        if not scores:
            return {
                "type": "silhouette_analysis",
                "scores": [],
                "optimal_k": 2,
                "error": "Could not calculate silhouette scores"
            }
        
        return {
            "type": "silhouette_analysis",
            "scores": scores,
            "optimal_k": max(scores, key=lambda x: x["score"])["n_clusters"]
        }
