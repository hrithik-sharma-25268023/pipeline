"""feature engineering module for data"""

import os
import pandas as pd


def age_category(age: float) -> str:
    """creates a age category"""

    if age >= 18 and age <= 25:
        return 'young-adults'
    elif age > 25 and age <= 40:
        return 'core-adults'
    elif age > 40 and age <= 60:
        return 'middle-aged-adults'
    elif age > 60 and age <= 100:
        return 'seniors'


def create_job_type(data: pd.DataFrame) -> pd.DataFrame:
    """creates a job type feature"""

    data['job-type'] = data['hours-per-week'].apply(lambda hour: 'part-time' if hour <= 20 else 'full-time')
    return data


def create_age_category(data: pd.DataFrame) -> pd.DataFrame:
    """creates a age category feature"""

    data['age-group'] = data['age'].apply(age_category)
    return data

if __name__ == "__main__":
    
    data_path = os.getcwd()+"/data/income.csv"
    data = pd.read_csv(data_path)
    data = create_job_type(data)
    data = create_age_category(data)
    data.to_csv(data_path, index=False)
