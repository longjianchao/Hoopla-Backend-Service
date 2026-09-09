from pydantic import BaseModel
from typing import List, Dict, Any

# Define the model component data structure
class ModelComponent(BaseModel):
    plane: str
    # Source plane fields
    size: float = None  # Optional field
    n_sersic: float = None  # Optional field
    # Lens plane fields
    theta_e: float = None  # Optional field
    # Shared fields
    ell: float
    ang: float
    x: float
    y: float

# Define the request data structure
class MCMCRequest(BaseModel):
    components: List[Dict[str, Any]]  # Use Dict to keep flexibility
    pixscale: float
    image_data: Any  # Image data may have a complex structure, so use the Any type
