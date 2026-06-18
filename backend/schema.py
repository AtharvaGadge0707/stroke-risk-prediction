from pydantic import BaseModel, Field
from typing import Literal


class PatientInput(BaseModel):
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    hypertension: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    ever_married: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    avg_glucose_level: float = Field(..., ge=0, le=300, description="Average glucose level in mg/dL")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index")
    gender: Literal["Male", "Female"] = Field(..., description="Patient gender")
    work_type: Literal["Govt_job", "Never_worked", "Private", "Self-employed", "children"] = Field(..., description="Type of work")
    Residence_type: Literal["Rural", "Urban"] = Field(..., description="Residence type")
    smoking_status: Literal["Unknown", "formerly smoked", "never smoked", "smokes"] = Field(..., description="Smoking status")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 67,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": 1,
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "gender": "Male",
                "work_type": "Private",
                "Residence_type": "Urban",
                "smoking_status": "formerly smoked"
            }
        }