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


class EnergyCertify(Enum):
    E = "E"
    D = "D"
    F = "F"
    C = "C"
    ISENTO_EM_TRAMITE = "Isento / Em Trâmite"
    A = "A"
    A_PLUS = "A+"
    B = "B"
    B_MINUS = "B-"
    G = "G"
    NA = "NA"
