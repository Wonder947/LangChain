# from https://python.langchain.com/en/latest/modules/models/llms/examples/custom_llm.html


# There is only one required thing that a custom LLM needs to implement:
# A _call method that takes in a string, some optional stop words, and returns a string
#
# There is a second optional thing it can implement:
# An _identifying_params property that is used to help with printing of this class. Should return a dictionary.


#Let’s implement a very simple custom LLM that just returns the first N characters of the input.
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}


# We can now use this as an any other LLM.
# n=10
n=20
llm = CustomLLM(n=n)
print(llm("count how many characters are in the output as related to n"))

# We can also print the LLM and see its custom print.
print(llm)


