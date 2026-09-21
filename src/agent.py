"""
agent.py — single entry point composing the full pipeline:
customer message -> intent + confidence -> confidence-gated retrieval ->
grounded reply -> response-safety checks -> escalation decision.

This is what the frontend and evaluation harness both call, so there is
exactly one place the end-to-end pipeline is wired together.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reply_generator import ReplyGenerator
from escalation_engine import decide


class Agent:
    def __init__(self):
        self.reply_generator = ReplyGenerator()

    def handle(self, customer_text, prior_context=None, k=3):
        prior_context = prior_context or []
        pipeline_result = self.reply_generator.generate(customer_text, prior_context, k=k)
        decision = decide(pipeline_result)
        return {**pipeline_result, "escalation_decision": decision}


if __name__ == "__main__":
    import json
    agent = Agent()
    print(f"Mock mode: {agent.reply_generator.mock_mode}\n")
    for msg in [
        "how do I get out of safe mode?",
        "my account was hacked and I lost items",
        "i tried everything and it still doesn't work, this is ridiculous!!",
        "thanks so much, appreciate the help",
    ]:
        out = agent.handle(msg)
        print(f"MSG: {msg}")
        print(f"  intent={out['predicted_intent']} conf={out['classifier_confidence']:.2f}")
        print(f"  action={out['escalation_decision']['action']} reasons={out['escalation_decision']['reason_codes']}")
        print(f"  reply={out['generated_reply'][:80]}")
        print()
