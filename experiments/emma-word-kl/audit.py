"""The policy wrapper disables the shared trainer's same-model 'independent' audit."""


class IndependentAudit:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Emma has no independent quality judge in this bounded experiment")
