import os
import asyncio
import pandas as pd
import logging
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from requests.exceptions import RequestException
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

from sqlalchemy.orm import Session
from app.models.dental_fee_schedule import DentalFeeSchedule
from app.models.fee_schedule import FeeSchedule
from app.models.medical_association import MedicalAssociation
from app.models.technology import Guidelines, MedicalAssessment, BillableItem
from app.services.gpt_service import GPTService

logger = logging.getLogger(__name__)

class MedicalAssessmentService:
    def __init__(self, gpt_service: GPTService, db: Session):
        self.gpt_service = gpt_service
        self.db = db
        self.selenium_url = os.getenv('SELENIUM_URL', 'http://chrome:4444/wd/hub')

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
        """Fetch and score medical guidelines using Selenium"""
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # Initialize remote driver
            driver = webdriver.Remote(
                command_executor=self.selenium_url,
                options=chrome_options
            )
            guidelines_list = []
            
            try:
                # Access guidelinecentral.com and get initial search results
                url = f"https://www.guidelinecentral.com/guidelines/{medical_association}"
                driver.get(url)
                
                # Wait for results to load
                wait = WebDriverWait(driver, 10)
                search_results = wait.until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "search-result"))
                )
                
                # Store guideline data before navigation
                guideline_data = []
                for result in search_results:
                    try:
                        meta = result.find_element(By.CLASS_NAME, "result-meta")
                        link_elem = meta.find_element(By.TAG_NAME, "a")
                        guideline_data.append({
                            'title': link_elem.text,
                            'link': link_elem.get_attribute("href")
                        })
                    except Exception as e:
                        logger.error(f"Error extracting guideline data: {e}")
                        continue
                
                print(f"Found {len(guideline_data)} search results for {medical_association}")
                
                # Process each guideline
                for data in guideline_data:
                    try:
                        title = data['title']
                        link = data['link']
                        print(f"Processing guideline: {title} - {link}")
                        
                        # Score relevance using GPT
                        score = float(await self.gpt_service._create_chat_completion(
                            system_prompt=(
                                f"Given this problem statement: {problem_statement} "
                                "rate on a scale of 0 to 1, predict how likely it is to contain "
                                "the current industry standard treatment for this ailment. "
                                "Return only the number."
                            ),
                            user_prompt=title,
                            temperature=0.0
                        ))
                        
                        # Get guideline content in a new page load
                        driver.get(link)
                        content_divs = wait.until(
                            EC.presence_of_all_elements_located((By.CLASS_NAME, "summary-item-body"))
                        )
                        content = " ".join(div.text for div in content_divs)
                        
                        guideline = Guidelines(
                            title=title,
                            link=link,
                            relevance_score=score,
                            content=content
                        )
                        guidelines_list.append(guideline)
                        
                        # Add delay between requests
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing guideline {title}: {e}")
                        continue
                        
                # Sort by relevance score and return top 3
                guidelines_list.sort(key=lambda x: x.relevance_score, reverse=True)
                return guidelines_list[:3]
                
            finally:
                driver.quit()

        except Exception as e:
            logger.error(f"Error fetching guidelines: {e}")
            raise

    async def _get_hcpcs_codes(self, recommendations: str) -> List[str]:
        """Extract HCPCS codes from recommendations"""
        system_prompt = (
            "Your are a medical domain expert. Given these medical recommendations, "
            "list out all of the HCPCS codes involved in such a prescription to help get the fees. "
            "List only the codes and nothing else. Output as a csv with no spaces between codes."
        )
        
        response = await self.gpt_service._create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=recommendations
        )
        
        # Split response into list of codes and return first 10
        codes = [code.strip() for code in response.split(',') if code.strip()]
        return codes[:10]

    async def _fetch_cms_fee(self, code: str) -> Dict:
        """Fetch fee data from CMS API for non-dental codes"""
        try:
            # CMS API endpoint URL with the HCPCS code filter
            api_url = f"https://data.cms.gov/data-api/v1/dataset/92396110-2aed-4d63-a6a2-5d6207d46a29/data?filter%5BHCPCS_Cd%5D={code}&size=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            # Extract data from the first record
                            record = data[0]
                            
                            # Get description from HCPCS_Desc field
                            description = record.get('HCPCS_Desc', 'No description available')
                            
                            # Get fee from Avg_Mdcr_Alowd_Amt field
                            fee_text = record.get('Avg_Mdcr_Alowd_Amt', '0')
                            
                            try:
                                # Parse the fee amount
                                fee = float(fee_text) if fee_text else 0.0
                                logger.info(f"Successfully fetched CMS fee for code {code}: ${fee}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to parse fee value '{fee_text}' for code {code}: {e}")
                                fee = 0.0
                            
                            return {
                                "code": code,
                                "description": description,
                                "fee": fee
                            }
                        else:
                            logger.warning(f"No fee data found for HCPCS code {code}")
                            return {
                                "code": code,
                                "description": "Fee data not found in CMS database",
                                "fee": 0.0
                            }
                    else:
                        logger.error(f"CMS API request failed with status {response.status} for code {code}")
                        return {
                            "code": code,
                            "description": f"API request failed with status {response.status}",
                            "fee": 0.0
                        }
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching CMS fee for code {code}: {e}")
            return {
                "code": code,
                "description": f"Network error: {str(e)}",
                "fee": 0.0
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching CMS fee for code {code}: {e}")
            return {
                "code": code,
                "description": f"Error fetching fee: {str(e)}",
                "fee": 0.0
            }

    async def calculate_fee_schedule(
        self,
        hcpcs_codes: List[str]
    ) -> List[Dict]:
        """Calculate fee schedule for HCPCS codes using both dental and CMS data"""
        try:
            fee_items = []
            
            for code in hcpcs_codes:
                # Handle dental codes (starting with "D")
                if code.startswith("D"):
                    # Query dental fee schedule from database
                    fee_item = self.db.query(DentalFeeSchedule)\
                        .filter(DentalFeeSchedule.code == code)\
                        .first()
                    
                    if fee_item:
                        fee_items.append({
                            "code": code,
                            "description": fee_item.description,
                            "fee": fee_item.average_fee,
                            "std_deviation": fee_item.std_deviation,
                            "percentile_50th": fee_item.percentile_50th, 
                            "percentile_75th": fee_item.percentile_75th,
                            "percentile_90th": fee_item.percentile_90th
                        })
                    else:
                        # If dental code not found
                        fee_items.append({
                            "code": code,
                            "description": "Dental code not found",
                            "fee": 0.0
                        })
                else:
                    # For non-dental codes
                    # First check if we have it in our database
                    medical_fee = self.db.query(FeeSchedule)\
                        .filter(FeeSchedule.hcpcs_code == code)\
                        .first()
                    
                    if medical_fee:
                        fee_items.append({
                            "code": code,
                            "description": medical_fee.description,
                            "fee": medical_fee.fee
                        })
                    else:
                        # Fetch from CMS website
                        fee_data = await self._fetch_cms_fee(code)
                        
                        # Add to database for future lookups
                        if fee_data["fee"] > 0:
                            new_fee = FeeSchedule(
                                hcpcs_code=code,
                                description=fee_data["description"],
                                fee=fee_data["fee"]
                            )
                            self.db.add(new_fee)
                            self.db.commit()
                        
                        fee_items.append(fee_data)
                
            logger.info(f"Calculated fees for {len(fee_items)} HCPCS codes")
            return fee_items

        except Exception as e:
            logger.error(f"Error calculating fee schedule: {e}")
            return [{"code": code, "description": "Error retrieving fee", "fee": 0.0} for code in hcpcs_codes]
    

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
            print(f"Guidelines found: {len(guidelines_list)}")
            # Add guidelines to assessment
            for guideline in guidelines_list:
                guideline.assessment_id = assessment.id
                self.db.add(guideline)
            
            # Step 3: Extract procedures from top guidelines
            all_guidelines_text = "\n\n".join(
                g.content for g in guidelines_list[:3] 
                if g.content is not None
            )
            print(f"Extracting procedures from guidelines: {all_guidelines_text[:100]}...")  # Log first 100 chars
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