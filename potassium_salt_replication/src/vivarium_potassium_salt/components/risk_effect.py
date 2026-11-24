"""A log-linear risk effect whose TMREL moves with the input draw."""
from __future__ import annotations

from typing import Any

from vivarium.config_tree import ConfigTree
from vivarium.engine.framework.engine import Builder
from vivarium.public_health.risks import RiskEffect

from vivarium_potassium_salt.constants import data_values


class SampledTmrelRiskEffect(RiskEffect):
    """The stock ``RiskEffect``, with the TMREL drawn anew for each input draw.

    The stock component multiplies a target rate by ``rr ** ((exposure - TMREL)
    / scale)``, floored at one, and puts the TMREL at the midpoint of the
    risk's theoretical-minimum-risk exposure distribution in every draw.  The
    paper, like GBD, treats that level as uncertain and draws it uniformly from
    the distribution's range (110 to 115 mmHg for systolic blood pressure) once
    per draw.  This subclass does the same: draw 0 keeps the midpoint, and draw
    ``k`` uses the level in ``ModelParameters.for_draw(k)``, the same one the
    artifact's attributable fractions were computed with.

    Everything else, including the component's name and configuration, is the
    stock component's.
    """

    def get_tmred(self, builder: Builder, configuration: ConfigTree | None = None) -> dict[str, Any]:
        tmred = super().get_tmred(builder, configuration)
        draw = builder.configuration.input_data.input_draw_number
        level = data_values.ModelParameters.for_draw(draw).tmrel
        if not tmred["min"] <= level <= tmred["max"]:
            raise ValueError(
                f"TMREL {level} for draw {draw} is outside the artifact's range "
                f"{tmred['min']} to {tmred['max']}"
            )
        # The stock component takes the midpoint of this range.
        return {**tmred, "min": level, "max": level}
