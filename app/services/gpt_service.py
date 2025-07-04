from typing import Dict, Tuple, Optional, List, Any
import os
import json
import pandas as pd
import logging
import asyncio
from functools import lru_cache
import time
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from app.core.config import settings
from sqlalchemy.orm import Session
from app.models.technology import Technology, ComparisonAxis, PatentSearch, PatentResult, MarketAnalysis
from app.services.patent_service import PatentService
from collections import OrderedDict
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncLRUCache:
    """Custom LRU cache for async functions"""
    def __init__(self, maxsize=100, ttl=3600):  # 1 hour TTL default
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl

    async def get_or_set(self, key, coroutine_func):
        now = datetime.now()
        
        # Check if key exists and is not expired
        if key in self.cache:
            value, timestamp = self.cache[key]
            if now - timestamp < timedelta(seconds=self.ttl):
                self.cache.move_to_end(key)
                return value
            else:
                del self.cache[key]

        # Generate new value
        value = await coroutine_func()
        
        # Add to cache
        self.cache[key] = (value, now)
        self.cache.move_to_end(key)
        
        # Remove oldest if cache is too large
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)
            
        return value

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    async def acquire(self):
        """Acquire permission to make an API call"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            # Wait until we can make another call
            await asyncio.sleep(60 - (now - self.calls[0]))
            
        self.calls.append(now)

class GPTService:
    """Service for interacting with OpenAI's GPT models and managing technology analysis."""
    
    def __init__(self, db: Session):
        """Initialize the GPT service with API key from settings."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4-turbo-preview"  # Using the latest model
        self.db = db
        self.rate_limiter = RateLimiter()
        self.patent_service = PatentService(db)
        
    async def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Get embedding vector for text using OpenAI's embedding model.
        
        Args:
            text: Text to get embedding for
            model: OpenAI embedding model to use
            
        Returns:
            List of floats representing the embedding vector
            
        Example:
            >>> embedding = await gpt_service.get_embedding("example text")
            >>> len(embedding)  # Should be 1536 for ada-002
            1536
        """
        try:
            await self.rate_limiter.acquire()
            text = text.replace("\n", " ")
            response = await self.client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []

    async def _create_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        retry_count: int = 3
    ) -> Optional[str]:
        """
        Centralized method for creating chat completions with retry logic.
        
        Args:
            system_prompt: The system message that sets the behavior
            user_prompt: The user's input message
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens in the response
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalize frequent tokens
            presence_penalty: Penalize repeated tokens
            retry_count: Number of retries on failure
            
        Returns:
            The generated text response or None if there's an error
            
        Example:
            >>> response = await gpt_service._create_chat_completion(
            ...     "You are a helpful assistant.",
            ...     "What is 2+2?"
            ... )
            >>> response
            '4'
        """
        for attempt in range(retry_count):
            try:
                await self.rate_limiter.acquire()
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                
                # Check if the response has content
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    return content.strip() if content else None
                else:
                    logger.warning("OpenAI returned empty response")
                    return None
            except RateLimitError:
                if attempt < retry_count - 1:
                    await asyncio.sleep(20 * (attempt + 1))  # Exponential backoff
                    continue
                logger.error("Rate limit reached after retries")
                return None
            except Exception as e:
                logger.error(f"Error in chat completion (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return None

    async def generate_problem_statement(self, technology_description: str) -> str:
        """
        Generate a problem statement from a technology description.
        
        Args:
            technology_description: Description of the technology to analyze
            
        Returns:
            A concise problem statement
        """
        system_prompt = "Given the description of this technology, identify what problem this technology is a solution to. Output this problem as a concise sentence."
        result = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=technology_description,
            temperature=1.0
        )
        return result or ""
    
    async def get_raw_axes(self, problem_statement: str) -> str:
        """
        Generate initial raw axes of comparison based on the problem statement.
        
        Args:
            problem_statement: The problem the technology aims to solve
            
        Returns:
            Raw axes as CSV string
        """
        system_prompt = ("Based on the defined problem, define the key conceptual components that would make up a solution to this problem. "
                        "For example: the Nintendo Switch's problem statement is to provide a uniquely differentiated gaming console that "
                        "consumers would choose over other consoles in the market. The axes of comparison are portability, innovativeness "
                        "of form factor, hardware capabilities, price point. Output these components as a csv")
        result = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=problem_statement,
            temperature=1.0
        )
        return result or ""

    async def refine_axes(self, raw_axes: str, num_axes: int) -> str:
        """
        Refine raw axes into a specific number of core comparison axes.
        
        Args:
            raw_axes: Raw axes as CSV string
            num_axes: Number of axes to generate
            
        Returns:
            Refined axes as CSV string with header
        """
        system_prompt = (f"simplify this list of conceptual components into {num_axes} core axes of comparison. No more, no less. "
                        "These axes will be used to evaluate the market viability of a given technology. Focus on specific facets "
                        "of technology. Only points of comparison will mainly be patent abstracts. Define both extremes of each axis; "
                        "Extreme1 should be the opposite of Extreme2. Output those axes of comparison as a csv in the form of "
                        "[Axis],[Extreme1],[Extreme2]. For example: Size,Big,Small. Do not include square brackets in generated axes. "
                        "The header is: Axis,Extreme1,Extreme2")
        result = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=raw_axes,
            temperature=0.0
        )
        if not result:
            return ""
        
        # Clean up any markdown or extra formatting
        cleaned_result = (
            result
            .replace("```csv", "")
            .replace("```", "")
            .strip()
        )
        
        # Validate CSV format
        lines = [line.strip() for line in cleaned_result.split('\n') if line.strip()]
        if not lines or not lines[0].lower().startswith("axis,extreme1,extreme2"):
            # If header is missing, add it
            lines.insert(0, "Axis,Extreme1,Extreme2")
        
        return "\n".join(lines)

    async def generate_comparison_axes(
        self, 
        technology_name: str,
        problem_statement: str,
        num_axes: int,
        technology_description: str,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate complete comparison axes analysis for a technology.
        
        Args:            
            technology_name: Name of the technology
            problem_statement: Problem statement for the technology
            data_dir: Base directory to store results
            num_axes: Number of comparison axes to generate (default: 3)
            
        Returns:
            Tuple containing:
            - DataFrame with comparison axes
            - Problem statement string
        """
        # Generate raw axes
        raw_axes = await self.get_raw_axes(problem_statement)
        if not raw_axes:
            return pd.DataFrame(), problem_statement

        # Refine axes
        axes_csv = await self.refine_axes(raw_axes, num_axes)
        if not axes_csv:
            return pd.DataFrame(), problem_statement

        try:
            # Clean the CSV string and split into lines
            lines = [line.strip() for line in axes_csv.split('\n') if line.strip()]
            
            if not lines:
                logger.error("No valid CSV lines found")
                return pd.DataFrame(), problem_statement
                
            # Parse header and data separately
            header = [col.strip() for col in lines[0].split(',')]
            data = []
            
            for line in lines[1:]:
                row = [cell.strip() for cell in line.split(',')]
                if len(row) == len(header):  # Only include rows that match header length
                    data.append(row)

            # Create DataFrame with explicit column names
            comp_axes = pd.DataFrame(data, columns=header)
            
            # Verify DataFrame structure
            required_cols = ['Axis', 'Extreme1', 'Extreme2']
            if not all(col in comp_axes.columns for col in required_cols):
                logger.error(f"Missing required columns. Found: {comp_axes.columns.tolist()}")
                return pd.DataFrame(), problem_statement

            # Create technology directory and save files
            tech_dir = os.path.join(os.getcwd(), "data", technology_name)
            os.makedirs(tech_dir, exist_ok=True)

            # Save comparison axes CSV
            csv_path = os.path.join(tech_dir, "comp_axes.csv")
            comp_axes.to_csv(csv_path, index=False)

            # Save metadata JSON
            metadata = {
                "name": technology_name,
                "Problem Statement": problem_statement,
                "Explanation": technology_description,
                "path": "comp_axes.csv"
            }
            with open(os.path.join(tech_dir, "meta_data.json"), "w") as f:
                json.dump(metadata, f, indent=4)

            return comp_axes, problem_statement

        except Exception as e:
            logger.error(f"Error creating comparison axes DataFrame: {e}")
            return pd.DataFrame(), problem_statement

    def create_technology(self, name: str, abstract: str, problem_statement: str, search_keywords: str = None) -> Technology:
        """Create a new technology entry in the database"""
        technology = Technology(
            name=name,
            abstract=abstract,
            problem_statement=problem_statement,
            search_keywords=search_keywords
        )
        self.db.add(technology)
        self.db.commit()
        self.db.refresh(technology)
        return technology

    def create_comparison_axes(self, technology_id: int, axes_data: List[dict]) -> List[ComparisonAxis]:
        """Create comparison axes for a technology
        
        Args:
            technology_id: ID of the technology
            axes_data: List of dicts containing axis_name, extreme1, extreme2, and weight
        """
        comparison_axes = []
        for axis in axes_data:
            comparison_axis = ComparisonAxis(
                technology_id=technology_id,
                axis_name=axis["axis_name"],
                extreme1=axis["extreme1"],
                extreme2=axis["extreme2"],
                weight=axis.get("weight", 1.0)  # Default weight of 1.0 if not specified
            )
            comparison_axes.append(comparison_axis)
        
        self.db.add_all(comparison_axes)
        self.db.commit()
        for axis in comparison_axes:
            self.db.refresh(axis)
        
        return comparison_axes

    def get_technology_by_id(self, technology_id: int) -> Optional[Technology]:
        """Retrieve a technology by its ID"""
        return self.db.query(Technology).filter(Technology.id == technology_id).first()

    def get_comparison_axes_by_technology(self, technology_id: int) -> List[ComparisonAxis]:
        """Get all comparison axes for a given technology"""
        return self.db.query(ComparisonAxis).filter(ComparisonAxis.technology_id == technology_id).all()

    async def process_and_save_analysis(
        self,
        technology_description: str,
        technology_name: str,
        data_dir: str,
        num_axes: int = 3
    ) -> Tuple[Technology, List[ComparisonAxis]]:
        """
        Process technology analysis and save to database
        
        Args:
            technology_description: Description of the technology
            technology_name: Name of the technology
            data_dir: Directory to store results
            num_axes: Number of comparison axes to generate
            
        Returns:
            Tuple of (Technology, List[ComparisonAxis])
        """
        # Generate analysis
        comp_axes_df, problem_statement = await self.generate_comparison_axes(
            technology_description=technology_description,
            technology_name=technology_name,
            data_dir=data_dir,
            num_axes=num_axes
        )
        
        if comp_axes_df.empty:
            return None, []

        # Generate search keywords
        search_keywords = await self.generate_search_keywords(problem_statement)

        # Create technology record
        technology = self.create_technology(
            name=technology_name,
            abstract=technology_description,
            problem_statement=problem_statement,
            search_keywords=search_keywords
        )

        # Convert DataFrame to format expected by create_comparison_axes
        axes_data = []
        for _, row in comp_axes_df.iterrows():
            axes_data.append({
                "axis_name": row["Axis"],
                "extreme1": row["Extreme1"],
                "extreme2": row["Extreme2"],
                "weight": 1.0  # Default weight
            })

        # Create comparison axes
        comparison_axes = self.create_comparison_axes(
            technology_id=technology.id,
            axes_data=axes_data
        )

        return technology, comparison_axes

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using their embeddings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
            
        Example:
            >>> score = await gpt_service.calculate_similarity(
            ...     "Machine learning model",
            ...     "AI algorithm"
            ... )
            >>> 0 <= score <= 1
            True
        """
        try:
            emb1 = await self.get_embedding(text1)
            emb2 = await self.get_embedding(text2)
            
            if not emb1 or not emb2:
                return 0.0
                
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5
            
            return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def get_search_term_score(self, word: str) -> float:
        """
        Score a word/phrase on how specific it is (0-1 scale).
        
        Args:
            word: Word or phrase to score
            
        Returns:
            Score between 0 and 1, where 0 is very generic and 1 is very specific
        """
        system_prompt = ("Score this word/phrase on a scale of 0 to 1 on how specific they would be. "
                        "0 is very generic. 1 is very specific. "
                        "Words like \"data\", \"machines\" and \"technology\" are generic while words like "
                        "\"PH level\", \"nanomachines\" and \"aircraft\" are more specific. "
                        "Do not include quotations, brackets, or explanations. Only return a number between 0 and 1.")
        
        result = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=word,
            temperature=0.0
        )
        try:
            return float(result) if result else 0.0
        except (ValueError, TypeError):
            return 0.0

    async def generate_search_keywords(
        self, 
        problem_statement: str,
        keyword_count: int = 2
    ) -> str:
        """
        Generate search keywords based on a problem statement.
        
        Args:
            problem_statement: The problem statement to generate keywords for
            keyword_count: Number of keywords to generate
            
        Returns:
            Space-separated string of keywords optimized for search
        """
        # Get problem statement embedding
        prob_embedding = await self.get_embedding(problem_statement)
        if not prob_embedding:
            return ""
            
        # Generate initial keywords with properly formatted strings
        system_prompt = (
            "You are a patent search specialist. Based on the given problem statement, "
            f"generate EXACTLY {keyword_count} search terms for patent search. \n\n"
            "Requirements:\n"
            f"1. Output MUST be exactly {keyword_count} comma-separated terms\n"
            "2. Each term MUST be a single word without any quotation marks\n"
            "3. Do not include numbers, quotations marks, or explanations\n"
            "4. Do not include generic terms like 'system' or 'device'\n"
            "5. Focus on specific technical terms\n\n"
            "Format your response as: term1,term2,term3,term4"
        )
        
        keyterms = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=problem_statement,
            temperature=0.9
        )
        
        if not keyterms:
            return ""
        
        print(f"Generated keyterms: {keyterms}")
        logger.info(f"Generated keyterms: {keyterms}")

        # Clean and validate keywords
        keywords = [k.strip().replace('"', '').replace("'", "") 
                for k in keyterms.split(",") 
                if k.strip()]
        
        # Ensure we only take the first keyword_count keywords
        keywords = keywords[:int(keyword_count)]
        print(f"Cleaned keywords: {keywords}")
        logger.info(f"Cleaned keywords: {keywords}")

        # Get embeddings and scores for each keyword
        keyword_data = []
        for word in keywords:
            keyword_embedding = await self.get_embedding(word)
            if not keyword_embedding:
                continue
                
            similarity = await self.calculate_similarity(problem_statement, word)
            specificity = await self.get_search_term_score(word)
            search_goodness = specificity * similarity
            
            keyword_data.append({
                "word": word,
                "search_goodness": search_goodness
            })
        
        if not keyword_data:
            return ""
        
        # Sort by search goodness
        keyword_data.sort(key=lambda x: x["search_goodness"], reverse=True)
        
        # Calculate average goodness
        avg_goodness = sum(k["search_goodness"] for k in keyword_data) / len(keyword_data)
        
        # Filter keywords above threshold and take top N
        good_keywords = [
            kw["word"] for kw in keyword_data 
            if kw["search_goodness"] > 0.5 * avg_goodness
        ][:int(keyword_count)]
        
        # If we don't have enough keywords after filtering, take top N from original list
        if len(good_keywords) < keyword_count:
            good_keywords = [kw["word"] for kw in keyword_data[:int(keyword_count)]]
        
        return " ".join(good_keywords)

    async def search_related_patents(self, technology_id: int) -> Optional[PatentSearch]:
        """
        Search for patents related to a technology using its search keywords
        
        Args:
            technology_id: ID of the technology to search patents for
            
        Returns:
            PatentSearch object containing results or None if error
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology or not technology.search_keywords:
            return None
            
        # Execute patent search
        return await self.patent_service.search_patents(
            technology_id=technology_id,
            search_query=technology.search_keywords
        )

    async def analyze_with_gpt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get structured analysis from GPT with score, explanation, and confidence.
        
        Args:
            system_prompt: The system message that sets up the analysis
            user_prompt: The content to analyze
            temperature: Controls randomness of the response
            
        Returns:
            Dictionary containing:
            - score: float between -1 and 1
            - explanation: string explaining the rating
            - confidence: float between 0 and 1
        """
        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            "Provide your response in JSON format with the following fields:\n"
            "- score: decimal number between -1 and 1 to two decimal places\n"
            "- explanation: brief explanation of your rating\n"
            "- confidence: decimal number between 0 and 1 indicating your confidence in the rating\n\n"
            "Example response format:\n"
            '{"score": 0.75, "explanation": "This technology shows strong...", "confidence": 0.85}'
        )
        
        try:
            response = await self._create_chat_completion(
                system_prompt=enhanced_system_prompt,
                user_prompt=user_prompt,
                temperature=temperature
            )
            
            if not response:
                return {"score": 0.0, "explanation": "", "confidence": 0.0}
                
            # Parse JSON response
            try:
                result = json.loads(response)
                # Validate and clean the response
                return {
                    "score": float(result.get("score", 0.0)),
                    "explanation": str(result.get("explanation", "")),
                    "confidence": float(result.get("confidence", 0.0))
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse GPT response as JSON: {response}")
                # Try to extract just the score if JSON parsing fails
                try:
                    score = float(response.strip())
                    return {
                        "score": score,
                        "explanation": "",
                        "confidence": 0.5
                    }
                except ValueError:
                    return {"score": 0.0, "explanation": "", "confidence": 0.0}
                    
        except Exception as e:
            logger.error(f"Error in analyze_with_gpt: {e}")
            return {"score": 0.0, "explanation": "", "confidence": 0.0}

    async def analyze_technology_on_axis(
        self,
        abstract: str,
        axis_name: str,
        extreme1: str,
        extreme2: str,
        problem_statement: str
    ) -> Dict[str, float]:
        """
        Analyze a technology on a specific comparison axis
        """
        system_prompt = (
            f"Given the abstract and explanation of a technology, rate the technology "
            f"on the axis of {axis_name} with -1 being closer to {extreme1} and 1 "
            f"being closer to {extreme2} when it comes to this technology's ability "
            f"to address this problem statement: {problem_statement}"
        )
        user_prompt = f"Abstract: {abstract}"
        
        response = await self.analyze_with_gpt(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        return {
            "score": float(response.get("score", 0.0)),
            "explanation": response.get("explanation", ""),
            "confidence": float(response.get("confidence", 0.0))
        }
    
    async def describe_pca_component(
        self,
        component_loadings: Dict[str, float],
        problem_statement: str
    ) -> str:
        """
        Generate a description of what a principal component represents based on its loadings
        """
        try:
            system_prompt = (
                "You are analyzing a principal component from a PCA analysis of technologies. "
                "Based on how different comparison axes contribute to this component, "
                "describe what this component might represent in simple terms.\n\n"
                f"Context - Problem Statement: {problem_statement}\n\n"
                "Provide a concise one-sentence description focusing on the strongest contributing factors."
            )

            # Convert and validate loadings
            numeric_loadings = {}
            for axis, value in component_loadings.items():
                try:
                    numeric_loadings[str(axis)] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid loading value for axis {axis}: {value}")
                    continue

            if not numeric_loadings:
                return "No valid component loadings available"

            # Sort loadings by absolute value
            sorted_loadings = sorted(
                numeric_loadings.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Format loadings string
            loadings_str = "\n".join([
                f"{axis}: {value:.3f}"
                for axis, value in sorted_loadings
            ])

            user_prompt = f"Component loadings:\n{loadings_str}"
            
            response = await self.analyze_with_gpt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            
            return response.get("explanation", "Description not available")
        
        except Exception as e:
            logger.error(f"Error describing PCA component: {str(e)}")
            return "Error generating component description"
    

    async def analyze_cluster(
        self,
        abstracts: List[str],
        problem_statement: str
    ) -> Dict[str, str]:
        """
        Generate name and description for a cluster of technologies
        """
        name_prompt = (
            "Given the abstracts of the following technologies, generate a name "
            "for the cluster based on similarities in technologies and methods "
            f"described and how they relate to the problem statement: {problem_statement}\n\n"
            "Return only the title."
        )
        
        description_prompt = (
            "Given the abstracts of the following technologies, generate a description "
            "for the cluster in 2 to 3 sentences explaining the common thread between "
            "these technologies and how they relate to the problem statement: "
            f"{problem_statement}"
        )
        
        abstracts_text = "\n\n".join(abstracts)
        
        # Use _create_chat_completion directly instead of analyze_with_gpt
        name = await self._create_chat_completion(
            system_prompt=name_prompt,
            user_prompt=abstracts_text,
            temperature=1.0
        )
        
        description = await self._create_chat_completion(
            system_prompt=description_prompt,
            user_prompt=abstracts_text,
            temperature=1.0
        )
        
        return {
            "name": name or "Unnamed Cluster",
            "description": description or "No description available"
        }
    
    async def generate_market_analysis_summary(self, technology_id: int) -> str:
        """
        Generate a summary insight of the technology's market analysis highlighting
        strengths and weaknesses compared to related patents.
        
        Args:
            technology_id: The ID of the technology to analyze
            
        Returns:
            A string containing the market analysis summary insight
            
        Example:
            "Overall, when benchmarked against related patents, this technology demonstrates 
            a significant competitive edge in 'Precision & Dexterity Enhancement' (average score: 0.85),
            consistently outperforming in areas requiring high accuracy. However, it faces its most 
            considerable challenges or shows the greatest potential for improvement in 'Cost-Effectiveness' 
            (average score: 0.20), where competing solutions may currently hold an advantage."
        """
        # Get the technology
        technology = self.db.query(Technology).filter(Technology.id == technology_id).first()
        if not technology:
            return "Technology not found"
            
        # Get the comparison axes
        comparison_axes = self.db.query(ComparisonAxis).filter(
            ComparisonAxis.technology_id == technology_id
        ).all()
        
        if not comparison_axes:
            return "No comparison axes found for this technology"
            
        # Create a mapping of axis_id to axis_name for later use
        axis_id_to_name = {axis.id: axis.axis_name for axis in comparison_axes}
        
        # Get market analysis data
        # Use the relationship to get market analyses for this technology
        market_analyses = technology.market_analyses
        if not market_analyses or len(market_analyses) == 0:
            return "No market analysis data available for this technology"
            
        # Calculate average scores for each axis
        axis_scores = {}
        for axis in comparison_axes:
            # Get all scores for this axis
            axis_scores[axis.id] = []
            
        # Populate scores
        for analysis in market_analyses:
            if analysis.axis_id in axis_scores:
                axis_scores[analysis.axis_id].append(analysis.score)
                
        # Calculate averages
        axis_averages = {}
        for axis_id, scores in axis_scores.items():
            if scores:
                axis_averages[axis_id] = sum(scores) / len(scores)
            else:
                axis_averages[axis_id] = 0
                
        # Find highest and lowest scoring axes
        if not axis_averages:
            return "Insufficient data to generate market analysis summary"
            
        highest_score_axis_id = max(axis_averages, key=axis_averages.get)
        lowest_score_axis_id = min(axis_averages, key=axis_averages.get)
        
        highest_score = axis_averages[highest_score_axis_id]
        lowest_score = axis_averages[lowest_score_axis_id]
        
        highest_axis_name = axis_id_to_name.get(highest_score_axis_id, "Unknown Axis")
        lowest_axis_name = axis_id_to_name.get(lowest_score_axis_id, "Unknown Axis")
        
        # Generate summary using GPT
        system_prompt = """
        You are an expert technology analyst providing concise, insightful summaries of market positioning.
        Based on the analysis data provided, create a brief, professional summary highlighting the technology's
        key strengths and weaknesses compared to related patents.
        
        Focus on the areas where it performs best and worst according to the comparison axes.
        Keep your response to 2-3 sentences maximum. Be specific about the axis names and scores.
        """
        
        user_prompt = f"""
        Technology Name: {technology.name}
        Technology Abstract: {technology.abstract}
        
        Comparison Axes Analysis:
        - Strongest Area: '{highest_axis_name}' (average score: {highest_score:.2f})
        - Weakest Area: '{lowest_axis_name}' (average score: {lowest_score:.2f})
        
        Please generate a concise market analysis summary highlighting these strengths and weaknesses.
        """
        
        try:
            summary = await self._create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            
            # Check if summary is None before calling strip()
            if summary is None:
                return "Unable to generate market analysis summary due to API issues"
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating market analysis summary: {str(e)}")
            return f"Error generating market analysis summary: {str(e)}"