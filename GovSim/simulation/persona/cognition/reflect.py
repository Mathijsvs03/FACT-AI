from simulation.utils import ModelWandbWrapper

from ..common import ChatObservation, PersonaIdentity
from .component import Component


class ReflectComponent(Component):

    prompt_insight_and_evidence: callable
    prompt_planning_thought_on_conversation: callable
    prompt_memorize_from_conversation: callable

    def __init__(self, model: ModelWandbWrapper):
        super().__init__(model)

    def run(self, focal_points: list[str]):
        acc = []
        for focal_point in focal_points:
            retireved_memory = self.persona.retrieve.retrieve([focal_point], 10)

            insights = self.prompt_insight_and_evidence(
                self.model, self.persona.identity, retireved_memory
            )
            for insight in insights:
                self.persona.store.store_thought(insight, self.persona.current_time)
                acc.append(insight)

    def reflect_on_convesation(self, conversation: list[tuple[str, str]], current_time=None):
        planning = self.prompt_planning_thought_on_conversation(
            self.model, self.persona.identity, conversation
        )  # TODO should be this be store in scratch for planning?
        memo = self.prompt_memorize_from_conversation(
            self.model, self.persona.identity, conversation
        )

        # HET GAAT IN DEZE FUNCTIE FOUT OMDAT NIET IEDEREEN EERST IN DE LOOP KOMT
        # TOEGEVOEGD:
        if hasattr(self.persona, 'current_time'):
            self.persona.store.store_thought(planning, self.persona.current_time)
            self.persona.store.store_thought(memo, self.persona.current_time)
        else:
            self.persona.store.store_thought(planning, current_time)
            self.persona.store.store_thought(memo, current_time)
        # TOEGEVOEGD:

