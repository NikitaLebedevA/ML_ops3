from pydantic import BaseModel, model_validator


class AddModelRequest(BaseModel):
    model_name: str
    hyperparameters: dict

    @model_validator(mode='before')
    def validate_data(cls, values):
        model_name = values.get('model_name')
        hyperparameters = values.get('hyperparameters')

        if not model_name:
            raise ValueError("Model name must be provided.")
        
        if not hyperparameters or not isinstance(hyperparameters, dict):
            raise ValueError("Hyperparameters must be provided as a dictionary.")
        
        if len(model_name) < 3:
            raise ValueError("Model name must be at least 3 characters long.")
        
        return values


class TrainingRequest(BaseModel):
    model_name: str
    dataset_key: str

    @model_validator(mode='before')
    def validate_data(cls, values):
        model_name = values.get('model_name')
        dataset_key = values.get('dataset_key')

        if not model_name:
            raise ValueError("Model name must be provided.")
        
        if not dataset_key:
            raise ValueError("Dataset key must be provided.")
        
        return values


class PredictionRequest(BaseModel):
    model_name: str
    input_data: list

    @model_validator(mode='before')
    def validate_data(cls, values):
        model_name = values.get('model_name')
        input_data = values.get('input_data')

        if not model_name:
            raise ValueError("Model name must be provided.")
        
        if not input_data or not isinstance(input_data, list):
            raise ValueError("Input data must be provided as a list.")
        
        return values


class DropModelRequest(BaseModel):
    model_name: str

    @model_validator(mode='before')
    def validate_data(cls, values):
        model_name = values.get('model_name')

        if not model_name:
            raise ValueError("Model name must be provided.")
        
        return values
