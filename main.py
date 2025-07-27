from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import cohere
import os
import random
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import csv
from io import StringIO
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import random
from gtts import gTTS
import os

# Load environment variables
env_path = Path('.') / 'variables.env'
load_dotenv(dotenv_path=env_path)

# Input Models with new fields for A/B testing
class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    job_title: str = Field(..., min_length=1, max_length=100)
    group: Literal["A", "B"] = "A"  # Default group for A/B testing

class Account(BaseModel):
    account_name: str = Field(..., min_length=1, max_length=200)
    industry: str = Field(..., min_length=1, max_length=100)
    pain_points: List[str] = Field(..., min_items=1, max_items=5)
    contacts: List[Contact] = Field(..., min_items=1)
    campaign_objective: Literal["awareness", "nurturing", "upselling"]

    # New fields for interest, tone, and language
    interest: str = Field(..., min_length=1, max_length=100)  # Personal interest input (e.g., movies, football, etc.)
    tone: Literal["formal", "casual", "enthusiastic", "neutral"] = "neutral"  # Tone of the email
    language: str = Field(..., min_length=1, max_length=200)  # Language for the email

class EmailVariant(BaseModel):
    subject: str
    body: str
    call_to_action: str
    sub_variants: List[str] = []  # List of alternative subject ideas

class Email(BaseModel):
    variants: List[EmailVariant]

class Campaign(BaseModel):
    account_name: str
    emails: List[Email]

class CampaignRequest(BaseModel):
    accounts: List[Account] = Field(..., min_items=1, max_items=10)
    number_of_emails: int = Field(..., gt=0, le=10)

class CampaignResponse(BaseModel):
    campaigns: List[Campaign]

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment on startup
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable is not set")
    yield

app = FastAPI(
    title="Email Drip Campaign API with A/B Testing",
    description="Generate personalized email campaigns with A/B testing using Cohere",
    version="1.0.0",
    lifespan=lifespan,
)

# Dependency for Cohere client
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="COHERE_API key not found")
    return cohere.Client(api_key)



def generate_email_content(client: cohere.Client, account: Account, email_number: int, total_emails: int, tone: str) -> List[EmailVariant]:
    """Generate a single email variant for the requested tone (instead of multiple variants)."""
    variants = []
    
    # Validate the tone input to ensure it's one of the allowed values
    if tone not in ["formal", "casual", "enthusiastic", "neutral"]:
        raise HTTPException(status_code=400, detail="Invalid tone provided. Must be one of: formal, casual, enthusiastic, neutral.")
    
    prompt = f"""
    Create a personalized email for the following business account:
    Company: {account.account_name}
    Industry: {account.industry}
    Pain Points: {', '.join(account.pain_points)}
    Campaign Stage: Email {email_number} of {total_emails}
    Campaign Objective: {account.campaign_objective}
    Recipient Job Title: {account.contacts[0].job_title}
    
    Interest: {account.interest}  # Personal interest field (e.g., movies, football)
    Tone: {tone}
    Language: {account.language}
    
    Generate a catchy and engaging subject line, personalized for the account and campaign objective. Please generate **three distinct subject lines** with different emphasis, tone, and phrasing, so that we can use one that best fits the campaign.

    Then, write the email body content with the following structure:
    1. An engaging email body personalized to the pain points and interest of the account
    2. A clear call-to-action encouraging the recipient to take the next step.
    3. Ensure the body is cohesive and flows well with the subject.

    Format the response as valid JSON with keys: "subject", "body", "call_to_action"
    """
    try:
        response = client.generate(
            model="command-xlarge-nightly",
            prompt=prompt,
            max_tokens=400,
            temperature=0.7,
        )
        
        # Log the response for debugging
        print(f"Response from Cohere API: {response}")
        
        # Extract the response and clean it
        response_text = response.generations[0].text.strip()

        # Safely parse the response as JSON
        try:
            email_data = json.loads(response_text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON response from Cohere")

        # Check for the subject field in the response
        if 'subject' not in email_data:
            raise HTTPException(status_code=500, detail="'subject' key missing in response from Cohere API")

        # Assuming Cohere will return the subjects in an array or list-like structure.
        sub_variants = [
            f"{email_data['subject'][0]}" ,
            f"{email_data['subject'][1]}",
            f"{email_data['subject'][2]}"
        ]
        
        # Create the EmailVariant object with the main subject and the sub-variants
        variants.append(
            EmailVariant(
                subject=email_data["subject"][0],  # Assuming first subject is chosen
                body=email_data["body"],
                call_to_action=email_data["call_to_action"],
                sub_variants=sub_variants  # Include sub-variants
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating email variant: {str(e)}"
        )

    return variants





def generate_campaign(client: cohere.Client, account: Account, number_of_emails: int) -> Campaign:
    """Generate a complete email campaign with A/B testing variants and sub-variants."""
    try:
        emails = []
        for contact in account.contacts:
            # Randomly assign groups for A/B testing
            contact.group = random.choice(["A", "B"])
        
        for i in range(number_of_emails):
            # Pass the tone from the account model or provide a default tone if it's not specified
            tone = account.tone if account.tone else "neutral"
            email_variants = generate_email_content(client, account, i + 1, number_of_emails, tone)
            emails.append(Email(variants=email_variants))
        
        return Campaign(account_name=account.account_name, emails=emails)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating campaign for {account.account_name}: {str(e)}"
        )

@app.post(
    "/generate-campaigns/",
    response_model=CampaignResponse,
    summary="Generate email campaigns with A/B testing",
    response_description="Generated email campaigns for the provided accounts"
)
def generate_campaigns(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
) -> CampaignResponse:
    """Generate personalized email campaigns for multiple accounts."""
    try:
        campaigns = []
        for account in request.accounts:
            campaign = generate_campaign(client, account, request.number_of_emails)
            campaigns.append(campaign)
        
        return CampaignResponse(campaigns=campaigns)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating campaigns: {str(e)}"
        )
from gtts import gTTS
from fastapi.responses import StreamingResponse
from io import BytesIO

# Function to generate text-to-speech from email content
def generate_tts_from_email(email_body: str, language: str = "en") -> StreamingResponse:
    """Converts email body text into speech and returns it as an audio file response."""
    try:
        tts = gTTS(text=email_body, lang=language, slow=False)
        
        # Save the speech to a BytesIO object (in-memory file)
        audio_file = BytesIO()
        tts.save(audio_file)
        
        # Move the cursor to the start of the file
        audio_file.seek(0)
        
        # Return the audio file as a streaming response
        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=email_audio.mp3"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/generate-email-audio/")
def generate_email_audio(email_body: str, language: str = "en"):
    """Endpoint to generate TTS from email content."""
    return generate_tts_from_email(email_body, language)


@app.post(
    "/export-campaigns-csv/",
    summary="Export campaigns as CSV",
    response_description="CSV file containing all generated campaigns"
)
def export_campaigns_csv(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
):
    """Export campaigns in CSV format for email automation tools."""
    try:
        # Generate the campaigns first
        campaigns_response = generate_campaigns(request, client)
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Account Name', 'Email Number', 'Variant', 'Subject', 'Sub-Variants', 'Body', 'Call to Action'])
        
        # Write data
        for campaign in campaigns_response.campaigns:
            for i, email in enumerate(campaign.emails, 1):
                for variant_idx, variant in enumerate(email.variants, 1):
                    writer.writerow([ 
                        campaign.account_name,
                        f"Email {i}",
                        f"Variant {variant_idx}",
                        variant.subject,
                        "; ".join(variant.sub_variants),  # Add sub-variants as a semicolon separated string
                        variant.body,
                        variant.call_to_action
                    ])
        
        # Prepare the response
        output.seek(0)
        filename = f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting campaigns to CSV: {str(e)}"
        )

@app.get(
    "/health",
    summary="Health check endpoint",
    response_description="Current API health status"
)
def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "cohere_api_configured": True
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
