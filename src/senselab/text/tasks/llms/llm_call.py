from openai import OpenAI
from typing import Dict, List, Optional
# from langchain_community.chat_models import ChatOpenAI # I had to run "pip install --only-binary :all: greenlet" first before installing langchain
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.output_parsers import StrOutputParser


# openrouter account associated with bruceatwood1@gmail.com
OPENROUTER_API_KEY = "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"


class llm_server:
    """
    Wrapper for invoking various LLMs.
    
    This class provides a unified interface for interacting with different large language models (LLMs).
    
    Parameters:
    -----------
    model : str
        The name of the model to use. This is a required parameter and should be one of the following options:
        
        - "mistral-7b"

    Attributes:
    -----------
    model : str
        The name of the selected model.

    Methods:
    --------
    invoke

    Example:
    --------
    To create an instance of llm_server with the "gpt-3.5-turbo" model:
    
    >>> llm = llm_server(model="mistral-7b")
    >>> response = llm.invoke(message = "say hello world", system_instruction = "add bumblebee on a new line on end", params)
    """
    
    def __init__(self, model_name: str):
        self._model_name = self._get_model(model_name)        
        self._client= OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key= OPENROUTER_API_KEY
        )


    def invoke(self, 
               message: str, 
               system_instruction: str, 
               params: Optional[Dict] = None) -> str:
        """
        Class method to invoke the model with a given message and system instruction.

        Parameters:
        -----------
        message : str
            The user message to send to the model.
        system_instruction : str
            The system instruction for the model.
        params : Optional[Dict]
            Additional parameters for the model invocation, if any.

        Returns:
        --------
            str
                The content of the model's response.
        """
        if params:
            for key, value in params.items():
                setattr(self._model, key, value)

        messages = [
            {
                "role": "user",
                "content": message,
            },
            {
                "role": "system",
                "content": system_instruction
            },
        ]
        
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
        )
        
        return completion.choices[0].message.content
    
    
    
    def _get_model(self, model):
        
        model_mapping = {
            "mistral_7b": "mistralai/mistral-7b-instruct:free"
        }
        if model in model_mapping:
            return model_mapping[model]
        else:
            available_options = ",\n\t".join(model_mapping.keys())
            raise ValueError(f"That is not a supported model. Available options: \n\t{available_options}")
