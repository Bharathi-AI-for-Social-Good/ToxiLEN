from models.main_model import MemeMultimodalDetector
from config import config

def build_model(variant: str):
   return MemeMultimodalDetector(
       config['predict'],
       variant=variant
       ).to(
           config['predict']['device']
           )