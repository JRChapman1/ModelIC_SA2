from modelic.products.product_types import ProductType
from modelic.products.annuity import Annuity
from modelic.products.endowment import Endowment
from modelic.products.pure_endowment import PureEndowment
from modelic.products.life_assurance import LifeAssurance


PRODUCT_FACTORY = {
    ProductType.Annuity: Annuity,
    ProductType.Endowment: Endowment,
    ProductType.PureEndowment: PureEndowment,
    ProductType.WholeOfLifeAssurance: LifeAssurance,
    ProductType.TermAssurance: LifeAssurance,
}