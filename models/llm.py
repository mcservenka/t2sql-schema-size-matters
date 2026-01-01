import os
import time
import json
from openai import OpenAI

TOOL_NAME = "t2sql_tool"
TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Process a natural language request and return an appropriate SQL SELECT query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "nullable": False,
                    "description": "SQL SELECT statement matching the user's intent."
                },
            }
        }
    }
}

class LLM:

    def __init__(self, provider:str = "openai", model:str = "gpt-5"):
        self.provider = provider
        self.model = model

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                organization=os.getenv('OPENAI_API_ORGANIZATION'),
                project=os.getenv('OPENAI_API_PROJECT'),
            )
        elif self.provider == "google":
            self.client = OpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        elif self.provider == "together":
            self.client = OpenAI(
                api_key=os.getenv("TOGETHERAI_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )
    
    # sending request to llm and receiving response
    def ask(self, messages):

        tool_to_use = TOOL

        start_time = time.perf_counter() # start timer

        chat_kwargs = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "tools": [tool_to_use],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}}
        }

        response = self.client.chat.completions.create(**chat_kwargs)

        end_time = time.perf_counter()  # end timer
        duration_seconds = end_time - start_time

        message = response.choices[0].message
        tool_call = message.tool_calls[0] if message.tool_calls else None
        
        tool_output = {}
        try:
            arguments = json.loads(tool_call.function.arguments)
            tool_output = {
                "sql": arguments.get("sql")
            }
        except Exception as e:
            print("Exception when deconstructing response")
            print(str(tool_call))
            tool_output = {
                "sql": None
            }      
        
        return {
            "response": tool_output,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "duration_seconds": duration_seconds,
        }
