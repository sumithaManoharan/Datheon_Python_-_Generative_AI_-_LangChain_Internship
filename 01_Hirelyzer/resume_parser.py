import  os, json, pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from base64 import b64encode
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pathlib import Path

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Experience(Strict):
    company : str
    job_title: str
    job_location: str
    start_date: str | None = Field(description="YYYY-MM, or null if not stated")
    end_date: str | None = Field(description="YYYY-MM, or null if the job is current")
    is_current: bool
    highlights: list[str] = Field(
        description="Key responsibilities and achievements for this job, one bullet per entry. "
                    "Stay close to the resume's wording and do not invent anything. "
                    "Empty list if the resume gives no details."
    )

class Education(Strict):
    institution: str | None
    degree: str | None = Field(description="example: B.Sc,M.Sc, MS, BE etc")
    field_of_study: str | None = Field(description=" actual field they got their masters or UG in example: maths, robotics, datascience and analytics etc ")
    start_year: str | None = Field(description="YYYY only. Null if not stated.")
    end_year: str | None = Field(description="YYYY only. Null if not stated.")
    grade: str | None = Field(description="Grade or GPA exactly as written, null if not stated")

class Projects(Strict):
    project_title: str
    project_description: str
    skills_used: list[str] | None = Field(description="skills or technologies used in the project")


class Resume(Strict):
    is_resume: bool = Field(description="true if its a resume, false if anything other than a resume")    
    language: str = Field(description="language of the resume")
    name: str
    headline: str
    email: str | None
    phone : str | None
    location: str | None
    skills: list[str] = Field(description="One skill per entry")
    experience: list[Experience]
    education: list[Education]
    certifications: list[str] = Field(description="certifications of the candidate")
    projects: list[Projects] | None 


load_dotenv(override=True)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
MODEL="openai/gpt-oss-20b"
DATA = "/home/sumitha_m/Datheon_Python_Generative_AI_LangChain_Internship/Datheon_Generative_AI_Internship/data"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)

with open("data/resumes/03_daniel_okafor_banner_header_chips.pdf", "rb") as f:
    document = b64encode(f.read()).decode()


file_path = Path("/home/sumitha_m/Datheon_Python_Generative_AI_LangChain_Internship/Datheon_Generative_AI_Internship/data/resumes/09_aisha_rahman_creative_skill_bars_photo.jpg")        
# pdf_path = Path(f"{DATA}{DATA}/resumes/02_emily_vargas_two_column_sidebar.pdf")  
resume_text = pymupdf4llm.to_markdown(str(file_path))


def extract(model, text):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You extract data from resumes. Use only what is written. "
                                          "If something is missing use null. Write dates as YYYY-MM."},
            {"role": "user", "content": f"Resume:\n\n{text}"},
        ],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "resume", "strict": True, "schema": Resume.model_json_schema()}},
        extra_body={"provider": {"require_parameters": True}},
    )
    raw = response.choices[0].message.content
    return Resume.model_validate_json(raw)      # the format check: raises ValidationError if wrong

result = extract(MODEL, resume_text)
print(result.model_dump_json(indent=2))