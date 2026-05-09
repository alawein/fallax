"""Validator prompt pack — 5 validators for reasoning evaluation."""


class ValidatorPack:
    """Builds validator prompts for evaluating model reasoning.

    Each method returns a fully-formed prompt string ready to send
    to a judge LLM. No LLM calls happen here — only prompt construction.
    """

    @staticmethod
    def reasoning_critic(prompt: str, answer: str, reasoning: str) -> str:
        """Build a reasoning critic prompt."""
        return (
            "You are a strict reasoning auditor.\n\n"
            f"Given:\nPROMPT: {prompt}\n"
            f"MODEL ANSWER: {answer}\n"
            f"REASONING: {reasoning}\n\n"
            "Task:\n"
            "1. Identify the FIRST step where reasoning may be invalid.\n"
            "2. Explain why it is invalid.\n"
            "3. If valid, explicitly confirm no issues.\n\n"
            "Be precise. Do not restate the solution."
        )

    @staticmethod
    def assumption_extractor(reasoning: str) -> str:
        """Build an assumption extractor prompt."""
        return (
            "Extract all implicit assumptions made in the reasoning.\n\n"
            f"REASONING: {reasoning}\n\n"
            "For each assumption:\n"
            "- State it clearly\n"
            "- Say whether it is justified by the prompt (YES/NO)"
        )

    @staticmethod
    def counterfactual_test(reasoning: str, perturbation: str) -> str:
        """Build a counterfactual stress test prompt."""
        return (
            "Given this reasoning:\n"
            f"{reasoning}\n\n"
            "Modify the input slightly:\n"
            f"{perturbation}\n\n"
            "Question: Does the reasoning still hold? "
            "If not, where does it break?"
        )

    @staticmethod
    def adversarial_challenger(reasoning: str) -> str:
        """Build an adversarial challenger prompt."""
        return (
            "Your goal is to prove the reasoning WRONG.\n\n"
            f"REASONING: {reasoning}\n\n"
            "Find any flaw, inconsistency, or unjustified step.\n\n"
            "If none exist, explain why the reasoning is robust."
        )

    @staticmethod
    def truth_judge(prompt: str, answer: str) -> str:
        """Build a final truth judge prompt."""
        return (
            "Given:\n"
            f"Prompt: {prompt}\n"
            f"Answer: {answer}\n\n"
            "Is the final answer correct? (YES/NO)\n\n"
            "Do NOT consider reasoning — only correctness."
        )
