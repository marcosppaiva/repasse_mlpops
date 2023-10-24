from pydantic import BaseModel

from enums import Condition, EnergyCertify, ResidenceType


class Imovel(BaseModel):
    district: str
    property_type: ResidenceType
    bathroom: int
    metric: float
    room: int
    energy_certify: EnergyCertify
    condition: Condition
