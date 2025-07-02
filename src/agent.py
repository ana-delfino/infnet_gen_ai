import getpass
import os

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
test = """
Estou pensando em me mudar para a Rua Haddock Lobo, 500, em São Paulo. 
O que tem por perto? Qual o tempo de deslocamento para a Av. Paulista, 1578, às 9h da manhã?
"""
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain, SequentialChain

from langchain.chat_models import init_chat_model

import logging

logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv 

load_dotenv()


class NeighborhoodTemplate:
    def __init__(self):
        self.system_template = """
        You are a real estate agent who helps customer discover interesting spots near a given addresses they are willing to live.
        
        The user's request will be denoted by four hashtags. Convert the
        user's request into a detailed nearby interesting spots.

        Try to include the specific address of each location.

        Remember to give small text with no more then 50 words about the neighborhood and always suggest 5 nearby interesting spots,
        and give them in a list.

        Return the nearby interesting spots as a bulleted list with clear start and end locations.
        Be sure to mention the address.
        If specific start and end locations are not given,
        choose ones that you think are suitable and give specific addresses.
        Your output must be small text and the list about the neighhood.
        Exemple:
        Estou pensando em me mudar para a Rua Sergipe, 446 - Consolação, São Paulo - SP. O que tem por perto? 
         O bairro é excelente, seguro tem uma delegacia perto e hospital, próximo ao metro, segue alguns lugares do bairro para você conhecer:
          - Le Blé - Casa de Pães : Morando nesse endereço você pode tomar café da manhã nessa excelente padaria.
        Endereço: R. Pará, 252 - Higienópolis, São Paulo - SP
         - Parque Buenos Aires: você pode passear no parque que fica bem perto
        Endereço: Av. Angélica, 1500 - Higienópolis, São Paulo - SP, 01227-000
        - Museu do Futebol: O museu do futebol também é um lugar que você deve conhecer no bairro. você vai adorar
        Endereço: Praça Charles Miller, s/n - Pacaembu, São Paulo - SP, 01234-010 
        - Sesc Consolação: Se você gosta de praticar esportes o sesc consolação fica muito perto
        Endereço: R. Dr. Vila Nova, 245 - Vila Buarque, São Paulo - SP, 01222-020
		- AB FAAP - Museu de Arte Brasileira: Se você gosta de ver um pouco de arte, então o museu de arte brasileira do bairro é maravilhoso
		Endereço: R. Alagoas, 903 - Higienópolis, São Paulo - SP, 01242-902
        """

        self.human_template = """
        #### {request}
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt,
                                                             self.human_message_prompt])
        
        
class MappingTemplate:
    def __init__(self):
        self.system_template = """
        You an agent system who converts detailed nearby points of interest
        into a list of coordinates

        The nearby points of interest will be denoted by four hashtags.
        Convert it into a list containing dictionaries with the latitude,
        longitude, address and name of each location.

        Retrieve a clean JSON object, no markdown notation.

        For example:

        ####
        nearby points of interest for Rua Sergipe, 446 - Consolação, São Paulo - SP 
        - Comment:  O bairro é seguro tem uma delegacia perto e hospital, próximo ao metro, segue alguns lugares do bairro para você conhecer:
        - spots:
          - Le Blé - Casa de Pães : Morando nesse endereço você pode tomar café da manhã nessa excelente padaria.
        Endereço: R. Pará, 252 - Higienópolis, São Paulo - SP
         - Parque Buenos Aires: você pode passear no parque que fica bem perto
        Endereço: Av. Angélica, 1500 - Higienópolis, São Paulo - SP, 01227-000
        - Museu do Futebol: O museu do futebol também é um lugar que você deve conhecer no bairro. você vai adorar
        Endereço: Praça Charles Miller, s/n - Pacaembu, São Paulo - SP, 01234-010 
        - Sesc Consolação: Se você gosta de praticar esportes o sesc consolação fica muito perto
        Endereço: R. Dr. Vila Nova, 245 - Vila Buarque, São Paulo - SP, 01222-020
		- AB FAAP - Museu de Arte Brasileira: Se você gosta de ver um pouco de arte, então o museu de arte brasileira do bairro é maravilhoso
		Endereço: R. Alagoas, 903 - Higienópolis, São Paulo - SP, 01242-902
        ####
        Output:
        {{
            "locations": [
                {{"lat":-23.548230904023505, "lon": -46.660014995262905, "address": "R. Pará, 252 - Higienópolis, São Paulo - SP, 01243-020", "name": "Le Blé - Casa de Pães "}},
	            {{"lat":-23.545856131546635, "lon":-46.6586800846597, "address": "Av. Angélica, 1500 - Higienópolis, São Paulo - SP, 01227-000", "name": "Parque Buenos Aires"}},
		        {{"lat":-23.547547837016033, "lon": -46.665289047620654, "address": "Praça Charles Miller, s/n - Pacaembu, São Paulo - SP, 01234-010 ", "name": "Museu do Futebol"}},
		        {{"lat":-23.54563638056293, "lon":-46.650008742351076, "address": "R. Dr. Vila Nova, 245 - Vila Buarque, São Paulo - SP, 01222-020", "name": "Sesc Consolação"}},
		        {{"lat":-23.545084103234558, "lon": -46.66267730466557, "address": "R. Alagoas, 903 - Higienópolis, São Paulo - SP, 01242-902", "name": "AB FAAP - Museu de Arte Brasileira"}}  
        """
        self.human_template = """
        #### {spots}
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt,
                                                             self.human_message_prompt])
   
class Agent:
    def __init__(self, google_api_key, model="gemini-2.0-flash", model_provider="google_genai"):
        self.google_api_key = google_api_key
        self.model = model
        self.model_provider = model_provider
        self.logger = logging.getLogger(__name__)
        self.chat_model = init_chat_model(model=self.model,
                                     model_provider = self.model_provider, google_api_key= self.google_api_key )

    def get_tips(self, request):
        travel_prompt = NeighborhoodTemplate()
        coordinates_prompt = MappingTemplate()
        
        parser = LLMChain(
            llm=self.chat_model,
            prompt=travel_prompt.chat_prompt,
            output_key="spots"
        )
        coordinates_converter = LLMChain(
            llm=self.chat_model,
            prompt= coordinates_prompt.chat_prompt,
            output_key="coordinates"
        )

        chain = SequentialChain(
            chains=[parser, coordinates_converter],
            input_variables=["request"],
            output_variables=["spots", "coordinates"],
            verbose=True
        )
        return chain(
            {"request": request},
            return_only_outputs=True
        )