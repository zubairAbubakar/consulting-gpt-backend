import logging
from datetime import datetime
from typing import Dict
from sqlalchemy.orm import Session

from app.models.technology import Recommendation
from app.services.gpt_service import GPTService

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self, gpt_service: GPTService, db: Session):
        self.gpt_service = gpt_service
        self.db = db

    async def generate_recommendations(
        self,
        technology_id: int,
        name: str,
        problem_statement: str,
        abstract: str,
        current_stage: str,
        market_data: Dict
    ) -> Dict:
        """Generate and save recommendations"""
        try:
            logger.info("Starting recommendation generation")
            
            # Generate recommendations
            general_assessment = await self._analyze_redundancy(
                problem_statement=problem_statement,
                abstract=abstract
            )

            logistical_analysis = await self._analyze_logistics(
                problem_statement=problem_statement,
                abstract=abstract,
                current_stage=current_stage
            )

            market_analysis = await self._analyze_market_position(
                name=name,
                market_data=market_data
            )

            # Create recommendation record
            recommendation = Recommendation(
                technology_id=technology_id,
                general_assessment=general_assessment,
                logistical_showstoppers=logistical_analysis,
                market_showstoppers=market_analysis,
                current_stage=current_stage
            )

            # Save to database
            self.db.add(recommendation)
            self.db.commit()
            self.db.refresh(recommendation)

            return {
                "id": recommendation.id,
                "general_assessment": general_assessment,
                "logistical_showstoppers": logistical_analysis,
                "market_showstoppers": market_analysis,
                "current_stage": current_stage,
                "created_at": recommendation.created_at.isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            self.db.rollback()
            raise

    async def _analyze_redundancy(self, problem_statement: str, abstract: str) -> str:
        """Analyze if solution is redundant"""
        system_prompt = (
            "You are a seasoned venture capitalist who has guided "
            "many startups towards successful exits"
        )
        user_prompt = (
            f"Problem to solve: {problem_statement}\n"
            f"Proposed solution: {abstract}\n\n"
            "Is this solution redundant to something that already exists? "
            "If so, explain why. If not, explain what makes it unique."
        )
        
        response = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )
        return response

    async def _analyze_logistics(
        self,
        problem_statement: str,
        abstract: str,
        current_stage: str
    ) -> str:
        """Analyze logistical requirements and showstoppers"""
        system_prompt = (
            "You are a seasoned venture capitalist who has guided "
            "many startups towards successful exits"
        )
        user_prompt = (
            f"Problem to solve: {problem_statement}\n"
            f"Proposed solution: {abstract}\n"
            f"Current stage of business development: {current_stage}\n\n"
            "Generate a list of critical items that MUST be handled at this "
            "current stage of business development or else the idea would fail. "
            "Focus on logistical and operational requirements."
        )
        
        response = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )
        return response

    async def _analyze_market_position(
        self,
        name: str,
        market_data: Dict
    ) -> str:
        """Analyze market position and requirements"""
        system_prompt = (
            "You are a seasoned venture capitalist who has guided "
            "many startups towards successful exits"
        )
        user_prompt = (
            f"Market analysis data: {market_data}\n"
            f"For product: {name}\n\n"
            "Generate a list of market-based requirements that MUST be "
            "handled or the product would fail in the market. Consider "
            "competitive positioning, market entry barriers, and customer needs."
        )
        
        response = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )
        return response