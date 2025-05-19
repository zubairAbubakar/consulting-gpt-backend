from sqlalchemy.orm import Session
from app.models import Technology, ComparisonAxis, RelatedTechnology
from app.db.database import SessionLocal
from app.models.technology import MarketAnalysis

def create_test_data():
    db = SessionLocal()
    try:
        # Create a main technology
        llm_tech = Technology(
            name="Large Language Models",
            abstract="Large Language Models (LLMs) are advanced AI models trained on vast amounts of text data.",
            problem_statement="How to effectively process and understand human language for various applications."
        )
        db.add(llm_tech)
        db.flush()  # Flush to get the ID

        # Create comparison axes
        axes = [
            ComparisonAxis(
                technology_id=llm_tech.id,
                axis_name="Resource Requirements",
                extreme1="Low Resource Usage",
                extreme2="High Resource Usage",
                weight=0.8
            ),
            ComparisonAxis(
                technology_id=llm_tech.id,
                axis_name="Accuracy",
                extreme1="Lower Accuracy",
                extreme2="Higher Accuracy",
                weight=1.0
            )
        ]
        db.add_all(axes)
        db.flush()

        # Create related technologies
        related_techs = [
            RelatedTechnology(
                technology_id=llm_tech.id,
                name="GPT-4",
                abstract="Advanced language model by OpenAI with strong reasoning capabilities",
                document_id="gpt4_paper",
                type="paper",
                url="https://openai.com/gpt-4"
            ),
            RelatedTechnology(
                technology_id=llm_tech.id,
                name="BERT",
                abstract="Bidirectional Encoder Representations from Transformers by Google",
                document_id="bert_paper",
                type="paper",
                url="https://arxiv.org/abs/1810.04805"
            )
        ]
        db.add_all(related_techs)
        db.flush()

        # Create Market analyses
        market_analyses = [
            MarketAnalysis(
                technology_id=llm_tech.id,
                related_technology_id=related_techs[0].id,  # GPT-4
                axis_id=axes[0].id,  # Resource Requirements
                score=0.9,
                explanation="GPT-4 requires significant computational resources for both training and inference",
                confidence=0.95
            ),
            MarketAnalysis(
                technology_id=llm_tech.id,
                related_technology_id=related_techs[0].id,  # GPT-4
                axis_id=axes[1].id,  # Accuracy
                score=0.95,
                explanation="GPT-4 demonstrates very high accuracy across various tasks",
                confidence=0.9
            ),
            MarketAnalysis(
                technology_id=llm_tech.id,
                related_technology_id=related_techs[1].id,  # BERT
                axis_id=axes[0].id,  # Resource Requirements
                score=0.6,
                explanation="BERT requires moderate computational resources",
                confidence=0.85
            ),
            MarketAnalysis(
                technology_id=llm_tech.id,
                related_technology_id=related_techs[1].id,  # BERT
                axis_id=axes[1].id,  # Accuracy
                score=0.75,
                explanation="BERT shows good accuracy but may not match latest models",
                confidence=0.85
            )
        ]
        db.add_all(market_analyses)
        
        # Commit all changes
        db.commit()
        print("Test data successfully created!")
        
        # Verify the data
        verify_data(db)
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        db.rollback()
    finally:
        db.close()

def verify_data(db: Session):
    # Verify technology
    tech = db.query(Technology).first()
    print(f"\nVerified Technology:")
    print(f"Name: {tech.name}")
    print(f"Abstract: {tech.abstract[:50]}...")
    
    # Verify comparison axes
    axes = db.query(ComparisonAxis).all()
    print(f"\nVerified Comparison Axes ({len(axes)}):")
    for axis in axes:
        print(f"- {axis.axis_name}: {axis.extreme1} vs {axis.extreme2}")
    
    # Verify related technologies
    related = db.query(RelatedTechnology).all()
    print(f"\nVerified Related Technologies ({len(related)}):")
    for rel in related:
        print(f"- {rel.name}: {rel.abstract[:50]}...")
    
    # Verify analysis results
    results = db.query(MarketAnalysis).all()
    print(f"\nVerified Market Analyses ({len(results)}):")
    for result in results:
        print(f"- Score {result.score:.2f} (Confidence: {result.confidence:.2f})")

if __name__ == "__main__":
    create_test_data()