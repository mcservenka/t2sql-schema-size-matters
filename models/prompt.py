import os

from models.llm import LLM


INIT_INSTRUCTION = (
    "You are a helpful assistant that processes natural language requests by returning SQLite queries. "
    "Always use the provided function tool to respond. Do not reply directly."
    "Your response must include the following attribute with value:\n"
    "- `sql`: The correct corresponding sqlite SELECT statement.\n"
    "Only refer to tables and fields defined in the schema. Do not guess. "
    "Do not return any answer unless using the function tool.\n"
)

class Prompter:

    def __init__(self, provider:str = "openai", model:str = "gpt-5.2", schema_string:str = None):
        self.provider = provider
        self.model = model
        self.llm = LLM(provider=self.provider, model=self.model)

        if schema_string:
            self.schema_string = schema_string
        else:
            raise ValueError("Schema string must not be empty!")
        
    def ask_question(self, question):

        messages = self._build_messages(question)
        response = self.llm.ask(messages=messages)

        return response


    def _build_messages(self, question):
        
        messages = [
            { "role": "system", "content": INIT_INSTRUCTION },
            { "role": "system", "content": self.schema_string },
            { "role": "user", "content": question }
        ]

        return messages


