from enum import Enum, auto


class ResidenceType(Enum):
    MORADIA = (auto(),)
    APARTAMENTO = auto()


class Condition(Enum):
    RUINA = "Ruína"
    NOVO = "Novo"
    RENOVADO = "Renovado"
    USADO = "Usado"
    EM_CONSTRUCAO = "Em construção"
    PARA_RECUPERAR = "Para recuperar"
