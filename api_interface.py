from abc import ABC, abstractmethod
from typing import Any, Tuple
from openai import OpenAI
import anthropic
import time


class LLMInterface(ABC):
    """
    Abstract base class for integrating Large Language Models (LLMs) into a competitive programming context.

    Attributes:
        prompt (str): Initial prompt to set the context for the LLM.
    """

    def __init__(self):
        """
        Initialize the LLMInterface with a predefined prompt for generating competitive programming solutions.
        """
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.
        """

    @abstractmethod
    def call_llm(user_prompt: str) -> Tuple[str, Any]:
        """
        Abstract method to interact with the LLM.

        Args:
            user_prompt (str): The prompt containing the problem statement for the LLM.

        Returns:
            Tuple[str, Any]: A tuple containing the generated solution and additional metadata.
        """
        pass

    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        """
        Generates a solution to a given competitive programming problem using the LLM.

        Args:
            problem_statement (str): The competitive programming problem statement.

        Returns:
            Tuple[str, Any]: The generated solution and associated metadata.
        """
        user_prompt = self.prompt + problem_statement
        response, meta = self.call_llm(user_prompt)
        return response, meta


class ExampleLLM(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.

    Attributes:
        client (OpenAI): Client instance for interacting with the OpenAI API.
    """

    def __init__(self):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        self.client = OpenAI()

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-4o model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, Any]: The LLM's response and metadata about the completion.
        """
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_prompt}],
        )
        return completion.choices[0].message.content, str(completion)


class ClaudeSonnet45(LLMInterface):
    """
    Claude Sonnet 4.5 with extended thinking at its maximum budget.

    Sonnet 4.5 caps output at 64K tokens, so the thinking budget is set just below that.
    Streaming is used because requests this long would otherwise hit HTTP timeouts.
    """

    MODEL = "claude-sonnet-4-5-20250929"
    MAX_TOKENS = 64000
    THINKING_BUDGET = 63999

    def __init__(self, system_prompt: str | None = None):
        """
        Args:
            system_prompt: optional system prompt (e.g. structured-reasoning instructions).
                None sends no system prompt, which is the original benchmark behavior.
        """
        super().__init__()
        self.client = anthropic.Anthropic(max_retries=5)
        self.system_prompt = system_prompt

    def request_kwargs(self, user_prompt: str) -> dict:
        kwargs = dict(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            thinking={"type": "enabled", "budget_tokens": self.THINKING_BUDGET},
            messages=[{"role": "user", "content": user_prompt}],
        )
        if self.system_prompt:
            kwargs["system"] = self.system_prompt
        return kwargs

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        start = time.monotonic()
        with self.client.messages.stream(**self.request_kwargs(user_prompt)) as stream:
            message = stream.get_final_message()
        latency = time.monotonic() - start
        # Only text blocks go to code extraction; thinking summaries are logged separately.
        text = "".join(b.text for b in message.content if b.type == "text")
        thinking = "\n\n".join(b.thinking for b in message.content if b.type == "thinking")
        meta = {
            "id": message.id,
            "model": message.model,
            "stop_reason": message.stop_reason,
            "usage": message.usage.model_dump(),
            "latency_s": round(latency, 1),
            "thinking_summary": thinking,
        }
        return text, meta


if __name__ == "__main__":
    """
    Example execution demonstrating how to use the ExampleLLM class to generate solutions.
    """
    llm = ExampleLLM()
    response, meta = llm.generate_solution("Hello world")
    print(response)
    print(meta)
