import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import json
import asyncio
import logging

from app.db.database import SessionLocal
from app.models.technology import ClusterMember, ClusterResult, MarketAnalysis, PCAResult, Technology, RelatedTechnology, RelatedPaper, PatentSearch, PatentResult, ComparisonAxis
from app.schemas.technology import TechnologyCreate, TechnologyRead, RelatedTechnologyRead
from app.services.gpt_service import GPTService
from app.services.patent_service import PatentService
from app.services.paper_service import PaperService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnologyService:
    def __init__(self, db: Session):
        self.db = db
        self.gpt_service = GPTService(db)
        self.patent_service = PatentService(db)
        self.paper_service = PaperService(db)
    
    async def create_technology(self, name: str, abstract: str = None, num_of_axes: int = 3) -> Technology:
        """
        Create a new technology record
        """
        try:
            # Create technology object first with known values
            db_technology = Technology(
                name=name,
                abstract=abstract,
                num_of_axes=num_of_axes
            )
            
            # Generate problem statement using the abstract
            if abstract:
                problem_statement = await self.gpt_service.generate_problem_statement(abstract)
                db_technology.problem_statement = problem_statement
            
            # Save to database
            self.db.add(db_technology)
            self.db.commit()
            self.db.refresh(db_technology)

            # Generate and save comparison axes
            axes_df, _ = await self.gpt_service.generate_comparison_axes(
                technology_name=name,
                problem_statement=problem_statement,
                num_axes=num_of_axes,
                technology_description=abstract
            )

            # Debug print to check DataFrame structure
            # print(f"DataFrame columns: {axes_df.columns.tolist()}")
            # print(f"DataFrame content:\n{axes_df}")

            # Save comparison axes to database
            if not axes_df.empty and 'Axis' in axes_df.columns:
                for _, row in axes_df.iterrows():
                    comparison_axis = ComparisonAxis(
                        technology_id=db_technology.id,
                        axis_name=row['Axis'],          # Use original column name
                        extreme1=row['Extreme1'],       # Use original column name
                        extreme2=row['Extreme2'],       # Use original column name
                        weight=1.0  # Default weight
                    )
                    self.db.add(comparison_axis)
                
                self.db.commit()
                self.db.refresh(db_technology)
            else:
                print("Error: DataFrame is empty or missing required columns")
            
            return db_technology
            
        except Exception as e:
            print(f"Error creating technology: {e}")
            self.db.rollback()
            return None
    
    def get_all_technologies(self) -> List[Technology]:
        """
        Get all technologies from the database
        """
        return self.db.query(Technology).all()
    
    def get_technology_by_id(self, technology_id: int) -> Optional[Technology]:
        """
        Get a technology by ID
        """
        return self.db.query(Technology).filter(Technology.id == technology_id).first()
    
    def get_related_technologies(self, technology_id: int) -> List[RelatedTechnology]:
        """
        Get related technologies for a specific technology
        """
        return self.db.query(RelatedTechnology).filter(
            RelatedTechnology.technology_id == technology_id
        ).all()
    
    async def generate_search_keywords(self, technology_id: int) -> bool:
        """
        Generate search keywords for a technology
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Generate keywords based on the technology name and problem statement
            keywords = await self.gpt_service.generate_search_keywords(                 
                technology.problem_statement,
                keyword_count=4
            )
            
            # Update the technology with the new keywords
            if keywords:
                technology.search_keywords = keywords
                self.db.commit()
                return True
            
            return False
        except Exception as e:
            print(f"Error generating search keywords: {e}")
            return False
    
    async def generate_comparison_axes(self, technology_id: int) -> bool:
        """
        Generate comparison axes for a technology
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Generate axes based on the technology name and problem statement
            axes_df = await self.gpt_service.generate_comparison_axes(
                technology.name,
                technology.problem_statement,
                technology.num_of_axes,
                technology.abstract
            )
            
            # Clear existing axes
            for axis in technology.comparison_axes:
                self.db.delete(axis)

            print(f"generated axes: {axes_df} ")
            # Save comparison axes to database
            for _, row in axes_df.iterrows():
                comparison_axis = ComparisonAxis(
                    technology_id=technology.id,
                    axis_name=row['axis_name'],
                    extreme1=row['extreme1'],
                    extreme2=row['extreme2'],
                    weight=row.get("weight", 1.0)
                )
                self.db.add(comparison_axis)
            
            self.db.commit()   

            return True
        except Exception as e:
            print(f"Error generating comparison axes: {e}")
            return False
    

    def get_comparison_axes(self, technology_id: int) -> List[ComparisonAxis]:
        """
        Get all comparison axes for a specific technology
        """
        return self.db.query(ComparisonAxis).filter(
            ComparisonAxis.technology_id == technology_id
        ).all()

    async def search_patents(self, technology_id: int) -> bool:
        """
        Search for patents related to a technology
        """
        try:
            # Get the technology
            technology = self.get_technology_by_id(technology_id)
            if not technology or not technology.search_keywords:
                print("No technology or search keywords found")
                return False

            print(f"Searching patents with keywords: {technology.search_keywords}")

            # Search patents using patent service first to ensure we have results
            search_results = await self.patent_service.search_patents(technology.id, technology.search_keywords)
            if not search_results:
                print("No patent results found")
                return False
            
            return True

        except Exception as e:
            print(f"Error searching patents: {e}")
            self.db.rollback()
            return False
    
    async def process_patent_results(self, technology_id: int) -> bool:
        """
        Process patent search results to create related technology entries
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Get the latest patent search for this technology
            patent_search = self.db.query(PatentSearch)\
                .filter(PatentSearch.technology_id == technology_id)\
                .order_by(PatentSearch.search_date.desc())\
                .first()
            
            if not patent_search:
                return False
                
            # Get the patent results for this search
            patent_results = self.db.query(PatentResult)\
                .filter(PatentResult.search_id == patent_search.id)\
                .all()
            
            print(f"Processing {len(patent_results)} patent results")
            
            # Process the results
            for result in patent_results:
                # Check if related technology already exists
                existing = self.db.query(RelatedTechnology)\
                    .filter(
                        RelatedTechnology.technology_id == technology_id,
                        RelatedTechnology.document_id == result.patent_id
                    ).first()
                
                if not existing:
                    # Create new related technology
                    patent_details = await self.patent_service.fetch_patent_details(result.patent_id)
                    if patent_details:
                        related_tech = RelatedTechnology(
                            technology_id=technology_id,
                            name=patent_details.get('title', ''),
                            abstract=patent_details.get('abstract', ''),
                            document_id=result.patent_id,
                            type="patent",
                            url=result.url,
                            publication_date=patent_details.get('publication_date', ''),
                            inventors=patent_details.get('inventors', ''),
                            assignees=patent_details.get('assignees', '')
                        )
                        self.db.add(related_tech)
                        print(f"Added new related technology: {related_tech.name}")
            
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error processing patent results: {e}")
            self.db.rollback()
            return False
        

    async def search_related_papers(self, technology_id: int) -> bool:
        """
        Search and store related papers for a technology with complete details
        """
        try:
            # Get the technology
            technology = self.get_technology_by_id(technology_id)
            if not technology or not technology.search_keywords:
                logger.error("No technology or search keywords found")
                return False
    
            # Search for papers
            paper_ids = await self.paper_service.search_papers(technology.search_keywords)
            if not paper_ids:
                logger.error("No papers found")
                return False
    
            papers_added = 0
            # Process each paper
            for paper_id in paper_ids:
                # Check if paper already exists
                existing = self.db.query(RelatedPaper).filter(
                    RelatedPaper.technology_id == technology_id,
                    RelatedPaper.paper_id == paper_id
                ).first()
    
                if not existing:
                    # Fetch paper details immediately
                    details = await self.paper_service.fetch_paper_details(paper_id)
                    if details:
                        # Create new paper entry with complete details
                        paper = RelatedPaper(
                            technology_id=technology_id,
                            paper_id=paper_id,
                            title=details["title"],
                            abstract=details["abstract"],
                            authors=details["authors"],
                            publication_date=details["publication_date"],
                            journal=details["journal"],
                            url=details["url"],
                            citation_count=details["citation_count"]
                        )
                        self.db.add(paper)
                        papers_added += 1
                        logger.info(f"Added paper: {details['title']}")
    
            if papers_added > 0:
                self.db.commit()
                logger.info(f"Added {papers_added} new papers")
            return True
    
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            self.db.rollback()
            return False        

    async def _get_primary_technology_scores(self, primary_technology: Technology, axis_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Helper to get or define scores for the primary technology against the given axes.
        """
    
        logger.info(f"Attempting to define scores for primary technology '{primary_technology.name}' against axes: {axis_names}")
        axes = primary_technology.comparison_axes;

        if not axes:
            logger.error("No comparison axes found")
            return None
        
        primary_scores = {}

        for axis in axes:
            try:
                # Perform analysis with safeguards
                abstract = primary_technology.abstract.strip() if primary_technology.abstract else primary_technology.name
                if len(abstract) < 10:  # Skip if abstract is too short
                    logger.warning(f"Skipping analysis for tech {primary_technology.id}: Abstract too short")
                    continue

                result = await self.gpt_service.analyze_technology_on_axis(
                    abstract=abstract,
                    axis_name=axis.axis_name,
                    extreme1=axis.extreme1,
                    extreme2=axis.extreme2,
                    problem_statement=primary_technology.problem_statement
                )
                if result and "score" in result:
                    primary_scores[axis.axis_name] = result.get("score", 0.0)

            except Exception as analysis_error:
                logger.error(f"Error analyzing primary technology {primary_technology.id} on axis {axis.axis_name}: {analysis_error}")
                continue  # Continue with next axis instead of failing completely

        return primary_scores        

    async def perform_market_analysis(self, technology_id: int) -> bool:
        """
        Perform market analysis for a technology across all comparison axes
        """
        try:
            # Get the technology and its components
            technology = self.get_technology_by_id(technology_id)
            if not technology:
                return False

            # Get comparison axes
            axes = technology.comparison_axes
            if not axes:
                logger.error("No comparison axes found")
                return False

            # Get related technologies (both patents and papers)
            related_techs = self.db.query(RelatedTechnology).filter(
                RelatedTechnology.technology_id == technology_id
            ).all()

            if not related_techs:
                logger.error("No related technologies found")
                return False

            # Analyze each technology across each axis
            for tech in related_techs:
                # Skip technologies without abstracts
                if not tech.abstract:
                    logger.warning(f"Skipping technology {tech.id} - {tech.name}: No abstract available")
                    continue

                for axis in axes:
                    # Check if analysis already exists
                    existing = self.db.query(MarketAnalysis).filter(
                        MarketAnalysis.technology_id == technology_id,
                        MarketAnalysis.related_technology_id == tech.id,
                        MarketAnalysis.axis_id == axis.id
                    ).first()

                    if not existing:
                        try:
                            # Perform analysis with safeguards
                            abstract = tech.abstract.strip() if tech.abstract else tech.name
                            if len(abstract) < 10:  # Skip if abstract is too short
                                logger.warning(f"Skipping analysis for tech {tech.id}: Abstract too short")
                                continue

                            result = await self.gpt_service.analyze_technology_on_axis(
                                abstract=abstract,
                                axis_name=axis.axis_name,
                                extreme1=axis.extreme1,
                                extreme2=axis.extreme2,
                                problem_statement=technology.problem_statement
                            )

                            if result and "score" in result:
                                # Save analysis
                                analysis = MarketAnalysis(
                                    technology_id=technology_id,
                                    related_technology_id=tech.id,
                                    axis_id=axis.id,
                                    score=result.get("score", 0.0),
                                    explanation=result.get("explanation", "No explanation provided"),
                                    confidence=result.get("confidence", 0.0)
                                )
                                self.db.add(analysis)
                                logger.info(f"Added analysis for tech {tech.id} on axis {axis.axis_name}")
                            else:
                                logger.warning(f"No valid analysis result for tech {tech.id} on axis {axis.axis_name}")

                        except Exception as analysis_error:
                            logger.error(f"Error analyzing tech {tech.id} on axis {axis.axis_name}: {analysis_error}")
                            continue  # Continue with next axis instead of failing completely

            self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error performing market analysis: {e}")
            self.db.rollback()
            return False


    async def perform_pca_analysis(self, technology_id: int) -> Optional[PCAResult]:
        try:
            primary_technology = self.db.query(Technology).get(technology_id)
            if not primary_technology:
                logger.error(f"Primary technology with ID {technology_id} not found.")
                return None
            if not primary_technology.name:
                logger.error(f"Primary technology ID {technology_id} has no name, cannot proceed.")
                return None

            analyses = self.db.query(MarketAnalysis).filter(
                MarketAnalysis.technology_id == technology_id
            ).all()

            # Organize data for related technologies
            data_dict = {}
            # tech_names_map = {} # Maps original ID (related_tech_id) to name for clarity
            all_axis_names = set() # To collect all unique axis names

            for analysis in analyses:
                related_tech = self.db.query(RelatedTechnology).get(analysis.related_technology_id)
                axis = self.db.query(ComparisonAxis).get(analysis.axis_id)

                if not related_tech or not related_tech.name or not axis or not axis.axis_name:
                    logger.warning(f"Skipping analysis entry due to missing related tech/axis details: analysis_id {analysis.id}")
                    continue

                tech_name = related_tech.name
                axis_name = axis.axis_name
                all_axis_names.add(axis_name)

                if tech_name not in data_dict:
                    data_dict[tech_name] = {}
                data_dict[tech_name][axis_name] = analysis.score
            
            # Convert to DataFrame for related technologies
            # Ensure all technologies have all axes, filling missing with 0
            df_related = pd.DataFrame.from_dict(data_dict, orient='index')
            df_related = df_related.reindex(columns=list(all_axis_names), fill_value=0)


            # --- Include Primary Technology ---
            primary_tech_scores = await self._get_primary_technology_scores(primary_technology, list(all_axis_names))
            
            df_combined = df_related # Start with related tech

            if primary_tech_scores:
                # Ensure primary_tech_scores keys (axis names) match all_axis_names
                # Fill any missing axes for primary tech with 0 as well
                primary_scores_full = {axis: primary_tech_scores.get(axis, 0) for axis in all_axis_names}
                
                primary_tech_series = pd.Series(primary_scores_full, name=primary_technology.name)
                
                if primary_technology.name in df_combined.index:
                    logger.warning(f"Primary technology name '{primary_technology.name}' conflicts with a related technology name. Appending '_PRIMARY'.")
                    df_combined.loc[f"{primary_technology.name}_PRIMARY"] = primary_tech_series
                else:
                    # df_combined = df_combined.append(primary_tech_series) # Old pandas
                    df_combined = pd.concat([df_combined, primary_tech_series.to_frame().T])


            if df_combined.empty:
                logger.error("PCA DataFrame is empty. Cannot perform PCA.")
                return None
            if df_combined.shape[0] < 2:
                 logger.warning(f"PCA DataFrame has only {df_combined.shape[0]} sample(s). PCA might not be meaningful or possible.")
                 # You might decide to return None or a simplified result here
                 if df_combined.shape[0] < 1 : return None # Definitely can't do PCA with 0 samples

            # Ensure there are features (columns)
            if df_combined.shape[1] == 0:
                logger.error("PCA DataFrame has no features (axes). Cannot perform PCA.")
                return None

            # Standardize the data
            scaler = StandardScaler()
            # fit_transform requires at least 1 sample.
            scaled_data = scaler.fit_transform(df_combined)

            # Perform PCA
            # n_components should be min(n_samples, n_features) if not specified, but we want 2D.
            # If n_samples or n_features is less than 2, PCA(n_components=2) will fail.
            n_samples = scaled_data.shape[0]
            n_features = scaled_data.shape[1]
            
            # We need at least 2 components for a 2D plot.
            # If n_features < 2, we can't get 2 principal components.
            # If n_samples < 2 (after ensuring n_features >=2), PCA might still be problematic.
            if n_features < 1: # Should have been caught by df_combined.shape[1] == 0
                 logger.error(f"Not enough features ({n_features}) for PCA.")
                 return None
            
            num_pca_components = min(2, n_samples, n_features)
            if num_pca_components < 1: # Should not happen if previous checks pass
                logger.error(f"Cannot determine valid number of PCA components ({num_pca_components}).")
                return None


            pca = PCA(n_components=num_pca_components)
            transformed_coordinates = pca.fit_transform(scaled_data)

            # Create results dictionary
            transformed_dict = {
                str(df_combined.index[i]): transformed_coordinates[i].tolist()
                for i in range(len(df_combined.index))
            }

            # Store component information
            components_data = []
            for i in range(pca.n_components_): # Iterate up to the actual number of components fitted
                loadings_array = pca.components_[i]
                axis_loadings = {
                    col: float(loading_val)
                    for col, loading_val in zip(df_combined.columns, loadings_array)
                }
                components_data.append({
                    "component_number": i + 1,
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                    "loadings": axis_loadings,
                    "description": f"Principal Component {i + 1}" # Placeholder
                })
            
            # If num_pca_components was 1, pad to have a second dummy component for 2D plot consistency if needed by frontend
            # Or, frontend should handle 1D PCA data if that's a valid scenario.
            # For now, assuming frontend expects 2D data or can handle fewer components.

            pca_result_db = PCAResult(
                technology_id=technology_id,
                components=components_data,
                transformed_data=transformed_dict,
                total_variance_explained=float(sum(pca.explained_variance_ratio_))
            )

            self.db.add(pca_result_db)
            self.db.commit()
            self.db.refresh(pca_result_db)

            logger.info(f"PCA analysis completed for '{primary_technology.name}'. Total variance explained: {pca_result_db.total_variance_explained:.2%}")
            
            # Trigger description generation (can be background)
            await self.describe_pca_components(pca_result_db, primary_technology)

            return pca_result_db

        except Exception as e:
            logger.error(f"Error in perform_pca_analysis for tech ID {technology_id}: {e}", exc_info=True)
            self.db.rollback()
            return None


    async def describe_pca_components(self, pca_result: PCAResult, technology: Technology) -> None:
        try:
            if not pca_result.components:
                logger.warning(f"No components found in PCA result ID {pca_result.id} to describe.")
                return

            updated_components_data = []
            for component_data_dict in pca_result.components: # component_data_dict is one element from the list
                loadings_dict = component_data_dict.get("loadings")
                component_num = component_data_dict.get("component_number")

                if not loadings_dict:
                    logger.warning(f"No loadings found for component {component_num} in PCA result ID {pca_result.id}. Skipping description.")
                    # Keep the original component data if loadings are missing
                    updated_components_data.append(component_data_dict)
                    continue
                
                description = await self.gpt_service.describe_pca_component(
                    loadings_dict, # This is already {axis_name: loading_value}
                    technology.problem_statement
                )
                
                # Create a new dict or update a copy to avoid modifying the iterated list item directly if it's complex
                new_component_data = component_data_dict.copy()
                new_component_data["description"] = description
                updated_components_data.append(new_component_data)
            
            # Update the components in the database object
            pca_result.components = updated_components_data
            self.db.add(pca_result) # Mark as changed
            self.db.commit()
            logger.info(f"Successfully described PCA components for PCA result ID {pca_result.id}")

        except Exception as e:
            logger.error(f"Error describing PCA components for PCA result ID {pca_result.id}: {e}", exc_info=True)
  

    # Add new background task function:
    async def describe_pca_components_background(self, technology_id: int, pca_result_id: int):
        """Background task to describe PCA components with its own db session"""
        db = SessionLocal()
        try:
            # Get fresh copies of objects with new session
            technology = db.query(Technology).get(technology_id)
            pca_result = db.query(PCAResult).get(pca_result_id)
            
            if not technology or not pca_result:
                logger.error("Could not find technology or PCA result for description")
                return
                
            service = TechnologyService(db)
            await service.describe_pca_components(pca_result, technology)
            
        except Exception as e:
            logger.error(f"Error describing PCA components: {e}")
        finally:
            db.close()

    async def perform_clustering(self, technology_id: int) -> bool:
        """
        Perform clustering analysis based on PCA results
        """
        try:
            # Get latest PCA result
            pca_result = self.db.query(PCAResult)\
                .filter(PCAResult.technology_id == technology_id)\
                .order_by(PCAResult.created_at.desc())\
                .first()
            
            if not pca_result:
                logger.error("No PCA results found for clustering")
                return False

            # Convert PCA data to numpy array for clustering
            tech_names = list(pca_result.transformed_data.keys())
            points = np.array([pca_result.transformed_data[name] for name in tech_names])
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(points)
            
            # Get the technology being analyzed
            technology = self.get_technology_by_id(technology_id)
            
            # Process each cluster
            for i in range(3):  # 3 clusters
                # Get points and technologies in this cluster
                cluster_mask = cluster_labels == i
                cluster_points = points[cluster_mask]
                cluster_tech_names = [tech_names[j] for j, mask in enumerate(cluster_mask) if mask]
                
                # Calculate cluster metrics
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - center, axis=1)
                spread = np.mean(distances)
                
                # Get abstracts for cluster description
                abstracts = []
                for tech_name in cluster_tech_names:
                    tech = self.db.query(RelatedTechnology)\
                        .filter(
                            RelatedTechnology.technology_id == technology_id,
                            RelatedTechnology.name == tech_name
                        ).first()
                    if tech and tech.abstract:
                        abstracts.append(f"{tech.name}: {tech.abstract}")
                
                # Generate cluster description using GPT
                cluster_info = await self.gpt_service.analyze_cluster(
                    abstracts=abstracts,
                    problem_statement=technology.problem_statement
                )
                
                # Create cluster result with enhanced metrics
                cluster = ClusterResult(
                    technology_id=technology_id,
                    name=cluster_info["name"],
                    description=cluster_info["description"],
                    contains_target=technology.name in cluster_tech_names,
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    cluster_spread=float(spread),
                    technology_count=len(cluster_points)
                )
                self.db.add(cluster)
                self.db.flush()  # Get cluster ID
                
                # Add cluster members with distance metrics
                for tech_name, point in zip(cluster_tech_names, cluster_points):
                    tech = self.db.query(RelatedTechnology)\
                        .filter(
                            RelatedTechnology.technology_id == technology_id,
                            RelatedTechnology.name == tech_name
                        ).first()
                    if tech:
                        distance = float(np.linalg.norm(point - center))
                        member = ClusterMember(
                            cluster_id=cluster.id,
                            technology_id=tech.id,
                            distance_to_center=distance
                        )
                        self.db.add(member)
            
            self.db.commit()
            logger.info(f"Clustering completed for technology {technology_id}")
            return True

        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            self.db.rollback()
            return False        