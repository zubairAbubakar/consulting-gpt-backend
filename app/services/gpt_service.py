from typing import Tuple, Optional, List, Any
import os
import json
import pandas as pd
import logging
import asyncio
from functools import lru_cache
import time
from openai import OpenAI, RateLimitError
from app.core.config import settings
from sqlalchemy.orm import Session
from app.models.technology import Technology, ComparisonAxis, PatentSearch, PatentResult
from app.services.patent_service import PatentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4-turbo-preview"  # Using the latest model
        self.db = db
        self.rate_limiter = RateLimiter()
        self.patent_service = PatentService(db)
        
    @lru_cache(maxsize=100)
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
                response = await self.client.chat.completions.create(
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
                return response.choices[0].message.content.strip()
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

    async def get_problem_statement(self, technology_description: str) -> str:
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
        return result or ""

    async def generate_comparison_axes(
        self, 
        technology_description: str,
        technology_name: str,
        data_dir: str,
        num_axes: int = 3
    ) -> Tuple[pd.DataFrame, str]:
        """
        Generate complete comparison axes analysis for a technology.
        
        Args:
            technology_description: Description of the technology to analyze
            technology_name: Name of the technology
            data_dir: Base directory to store results
            num_axes: Number of comparison axes to generate (default: 3)
            
        Returns:
            Tuple containing:
            - DataFrame with comparison axes
            - Problem statement string
        """
        # Generate problem statement
        problem_statement = await self.get_problem_statement(technology_description)
        if not problem_statement:
            return pd.DataFrame(), ""

        # Generate raw axes
        raw_axes = await self.get_raw_axes(problem_statement)
        if not raw_axes:
            return pd.DataFrame(), problem_statement

        # Refine axes
        axes_csv = await self.refine_axes(raw_axes, num_axes)
        if not axes_csv:
            return pd.DataFrame(), problem_statement

        # Parse CSV into DataFrame
        axes_rows = axes_csv.split("\n")
        cols = axes_rows[0].split(",")  # Header row
        data = [row.split(",") for row in axes_rows[1:]]
        comp_axes = pd.DataFrame(data, columns=cols)

        # Create technology directory and save files
        tech_dir = os.path.join(data_dir, technology_name)
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
        search_keywords = await self.get_search_keywords(problem_statement)

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

    async def get_search_keywords(
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
            
        # Generate initial keywords
        system_prompt = (f"Based on this problem statement in user prompt, come up with {keyword_count} "
                      "single word search terms to find potential competing patented technologies. "
                      "Output as csv list. Each element should be one word. Do not include quotations.")
        
        keyterms = await self._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=problem_statement,
            temperature=0.9
        )
        
        if not keyterms:
            return ""
            
        keywords = [k.strip() for k in keyterms.split(",")]
        
        # Get embeddings and scores for each keyword
        keyword_data = []
        for word in keywords:
            # Get embedding and calculate similarity with problem statement
            keyword_embedding = await self.get_embedding(word)
            if not keyword_embedding:
                continue
                
            # Calculate cosine similarity with problem statement
            similarity = await self.calculate_similarity(problem_statement, word)
            
            # Get specificity score
            specificity = await self.get_search_term_score(word)
            
            # Calculate search goodness score - combining both similarity and specificity
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
        
        # Build final search term from keywords above 50% of average goodness
        search_terms = []
        for kw in keyword_data:
            if kw["search_goodness"] > 0.5 * avg_goodness:
                search_terms.append(kw["word"])
                
        return " ".join(search_terms)

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
