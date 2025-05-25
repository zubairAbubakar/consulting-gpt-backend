import asyncio
import os
import pandas as pd
import logging
from selenium import webdriver
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple

from sqlalchemy.orm import Session
from app.models.fee_schedule import FeeSchedule
from app.models.medical_association import MedicalAssociation
from app.models.technology import Guidelines, MedicalAssessment, BillableItem
from app.services.gpt_service import GPTService

logger = logging.getLogger(__name__)

class MedicalAssessmentService:
    def __init__(self, gpt_service: GPTService, db: Session):
        self.gpt_service = gpt_service
        self.db = db

    async def classify_medical_association(self, problem_statement: str) -> str:
        """
        Identify relevant medical association for a problem statement
        Returns the acronym of the most relevant medical association
        """
        try:
            # Get all medical associations
            associations = self.db.query(MedicalAssociation).all()
            
            # Create prompt for GPT
            system_prompt = (
                "You are a medical domain expert. Given a medical problem statement "
                "and a list of medical associations, identify the most relevant association. "
                "Return only the acronym of the most relevant association."
            )
            
            # Format associations for prompt
            associations_text = "\n".join([
                f"{assoc.acronym}: {assoc.organization}"
                for assoc in associations
            ])
            
            user_prompt = (
                f"Problem Statement: {problem_statement}\n\n"
                f"Available Medical Associations:\n{associations_text}\n\n"
                "Return only the acronym of the most relevant association."
            )
            
            # Get response from GPT
            acronym = await self.gpt_service._create_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0
            )
            print(f"GPT Response: {acronym}")
            # Validate acronym exists
            association = self.db.query(MedicalAssociation)\
                .filter(MedicalAssociation.acronym == acronym.strip())\
                .first()
            
            if not association:
                return "No medical associations found"
                
            return association.acronym

        except Exception as e:
            logger.error(f"Error classifying medical association: {e}")
            return "No medical associations found"

    async def get_medical_guidelines(
        self,
        medical_association: str,
        technology_name: str,
        problem_statement: str
    ) -> List[Guidelines]:
        """Fetch and score medical guidelines from guidelinecentral.com"""
        try:
            # Setup Chrome driver
            driver = webdriver.Chrome()
            
            # Access guidelinecentral.com with medical association
            url = f"https://www.guidelinecentral.com/guidelines/{medical_association}"
            driver.get(url)
            await asyncio.sleep(2)
            
            # Get page content
            page_source = driver.page_source
            driver.quit()
            
            # Parse content
            soup = BeautifulSoup(page_source, 'html.parser')
            search_results = soup.findAll('div', {'class': 'search-result'})
            
            guidelines_list = []
            
            # Create prompt for scoring relevance
            system_prompt = (
                f"Given this problem statement: {problem_statement} "
                "rate on a scale of 0 to 1, predict how likely it is to contain "
                "the current industry standard treatment for this ailment. "
                "Return only the number. Do not add any additional text or explanation."
            )
            
            # Process each search result
            for result in search_results:
                try:
                    meta = result.find('div', {'class': 'result-meta'})
                    title = meta.find('a').text
                    link = meta.find('a')['href']
                    
                    # Score relevance using GPT
                    score = float(await self.gpt_service._create_chat_completion(
                        system_prompt=system_prompt,
                        user_prompt=title,
                        temperature=0.0
                    ))
                    
                    guideline = Guidelines(
                        title=title,
                        link=link,
                        relevance_score=score
                    )
                    guidelines_list.append(guideline)
                    
                except Exception as e:
                    logger.error(f"Error processing guideline result: {e}")
                    continue
            
            # Sort by relevance score
            guidelines_list.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Get content for top 3 guidelines
            for guideline in guidelines_list[:3]:
                try:
                    driver = webdriver.Chrome()
                    driver.get(guideline.link)
                    await asyncio.sleep(1.5)
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    driver.quit()
                    
                    content_divs = soup.findAll('div', {'class': 'summary-item-body'})
                    content = " ".join(div.text for div in content_divs)
                    guideline.content = content
                    
                except Exception as e:
                    logger.error(f"Error fetching guideline content: {e}")
                    continue
            
            return guidelines_list

        except Exception as e:
            logger.error(f"Error fetching guidelines: {e}")
            raise

    async def _get_hcpcs_codes(self, recommendations: str) -> List[str]:
        """Extract HCPCS codes from recommendations"""
        system_prompt = (
            "Given these medical recommendations, list out all of the HCPCS codes involved. "
            "List only the codes and nothing else. Output as a csv with no spaces between codes."
        )
        
        response = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=recommendations
        )
        
        # Split response into list of codes
        codes = [code.strip() for code in response.split(',') if code.strip()]
        return codes

    async def calculate_fee_schedule(
        self,
        hcpcs_codes: List[str]
    ) -> List[Dict]:
        """Calculate fee schedule for HCPCS codes"""
        try:
            # Query fee schedule from database
            fee_items = []
            
            # Load fee schedule from database
            fee_schedule = self.db.query(FeeSchedule)\
                .filter(FeeSchedule.hcpcs_code.in_(hcpcs_codes))\
                .all()
            
            for code in hcpcs_codes:
                fee_item = next(
                    (item for item in fee_schedule if item.hcpcs_code == code),
                    None
                )
                
                if fee_item:
                    fee_items.append({
                        "code": code,
                        "description": fee_item.description,
                        "fee": fee_item.fee
                    })
                else:
                    # If code not found, add with placeholder values
                    fee_items.append({
                        "code": code,
                        "description": "Description not found",
                        "fee": 0.0
                    })
            
            return fee_items

        except Exception as e:
            logger.error(f"Error calculating fee schedule: {e}")
            raise

    async def extract_procedures(
        self,
        guidelines: str,
        problem_statement: str
    ) -> Tuple[List[str], str]:
        """Extract billable procedures and HCPCS codes"""
        system_prompt = (
            "Based on these treatment guidelines, what billable activities/procedures "
            "would a medical professional prescribe? Answer in how a medical professional would communicate to their billing people. " 
            "Return as a short list. Do not restate the question. Counseling is not a billable activity."
        )
        recommendations = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=f"Guidelines: {guidelines}\nProblem: {problem_statement}"
        )

        # Extract HCPCS codes
        codes = await self._get_hcpcs_codes(recommendations)
        
        return codes, recommendations

    async def create_medical_assessment(
        self,
        technology_id: int,
        problem_statement: str,
        technology_name: str
    ) -> MedicalAssessment:
        """Create complete medical assessment"""
        try:
            # Step 1: Classify medical association
            medical_association = await self.classify_medical_association(problem_statement)
            
            if medical_association == "No medical associations found":
                return None

            # Create assessment record
            assessment = MedicalAssessment(
                technology_id=technology_id,
                medical_association=medical_association
            )
            self.db.add(assessment)
            self.db.flush()  # Get ID without committing

            # Step 2: Get guidelines
            guidelines_list = await self.get_medical_guidelines(
                medical_association,
                technology_name,
                problem_statement
            )
            
            # Add guidelines to assessment
            for guideline in guidelines_list:
                guideline.assessment_id = assessment.id
                self.db.add(guideline)
            
            # Step 3: Extract procedures from top guidelines
            all_guidelines_text = "\n\n".join(
                g.content for g in guidelines_list[:3] 
                if g.content is not None
            )
            
            codes, recommendations = await self.extract_procedures(
                all_guidelines_text, 
                problem_statement
            )

            # Update assessment with recommendations
            assessment.recommendations = recommendations

            # Step 4: Calculate fees
            fee_schedule = await self.calculate_fee_schedule(codes)

            # Add billable items
            for item in fee_schedule:
                billable_item = BillableItem(
                    assessment_id=assessment.id,
                    hcpcs_code=item["code"],
                    description=item["description"],
                    fee=item["fee"]
                )
                self.db.add(billable_item)

            self.db.commit()
            self.db.refresh(assessment)
            
            return assessment

        except Exception as e:
            logger.error(f"Error in medical assessment: {e}")
            self.db.rollback()
            raise