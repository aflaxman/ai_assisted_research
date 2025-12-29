
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from laser.generic.shared import State
from laser.generic.components import TransmissionSI, nb_timer_update_timer_set
from laser.generic.utils import validate

# Define custom states
STATE_CARCASS = 4
STATE_REMOVED = 5

class SporeEnvironment:
    """
    Manages environmental spores.
    Updates spores[tick+1] based on spores[tick], decay, and new additions.
    """
    def __init__(self, model, decay_rate=0.0):
        self.model = model
        self.decay_rate = decay_rate

        # Add properties if they don't exist
        if not hasattr(self.model.nodes, 'spores'):
             self.model.nodes.add_vector_property("spores", model.params.nticks + 1, dtype=np.float32)
        if not hasattr(self.model.nodes, 'daily_spores'):
             self.model.nodes.add_vector_property("daily_spores", model.params.nticks + 1, dtype=np.float32)

    def step(self, tick):
        # Update spores for next tick
        # spores[t+1] = spores[t] * (1 - decay) + daily_spores[t]
        current = self.model.nodes.spores[tick]
        new_additions = self.model.nodes.daily_spores[tick]

        remaining = current * (1.0 - self.decay_rate)
        self.model.nodes.spores[tick + 1] = remaining + new_additions

class EnvironmentalTransmission:
    """
    S -> I based on environmental spores.
    Force of infection lambda = beta * spores
    """
    def __init__(self, model, beta, infdurdist, infdurmin=1):
        self.model = model
        self.beta = beta
        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("newly_infected", model.params.nticks + 1, dtype=np.int32)
        if not hasattr(self.model.nodes, 'spores'):
             self.model.nodes.add_vector_property("spores", model.params.nticks + 1, dtype=np.float32)

    def step(self, tick):
        spores = self.model.nodes.spores[tick]

        # Force of infection
        # We treat spores as a concentration or proxy for risk.
        # simple linear relationship
        ft = self.beta * spores

        # Convert to probability
        ft = -np.expm1(-ft)

        # Use TransmissionSI's helper to apply infection
        newly_infected_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)

        TransmissionSI.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            ft,
            newly_infected_by_node,
            self.model.people.itimer,
            self.infdurdist,
            self.infdurmin,
            tick,
        )

        newly_infected_by_node = newly_infected_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)

        # Update counts
        self.model.nodes.S[tick + 1] -= newly_infected_by_node
        self.model.nodes.I[tick + 1] += newly_infected_by_node
        self.model.nodes.newly_infected[tick] = newly_infected_by_node

class InfectionToCarcass:
    """
    I -> Carcass
    Uses itimer.
    """
    def __init__(self, model, decomp_dist, decomp_min=1):
        self.model = model
        self.decomp_dist = decomp_dist
        self.decomp_min = decomp_min

        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
        self.model.people.add_scalar_property("ctimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("C", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("newly_carcass", model.params.nticks + 1, dtype=np.int32)

    def step(self, tick):
        newly_carcass_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)

        # We need a helper that transitions state AND sets a new timer (ctimer)
        # laser.generic.components.nb_timer_update_timer_set does exactly this.

        nb_timer_update_timer_set(
            self.model.people.state,
            State.INFECTIOUS.value,
            self.model.people.itimer,
            STATE_CARCASS,
            self.model.people.ctimer,
            newly_carcass_by_node,
            self.model.people.nodeid,
            self.decomp_dist,
            self.decomp_min,
            tick
        )

        newly_carcass_by_node = newly_carcass_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)

        self.model.nodes.I[tick + 1] -= newly_carcass_by_node
        self.model.nodes.C[tick + 1] += newly_carcass_by_node
        self.model.nodes.newly_carcass[tick] = newly_carcass_by_node

@nb.njit(nogil=True, parallel=True)
def nb_carcass_step(states, ctimers, nodeids, scavenging_rate, spores_per_carcass, daily_spores, newly_removed_by_node, newly_scavenged_by_node):
    for i in nb.prange(len(states)):
        if states[i] == STATE_CARCASS:
            nid = nodeids[i]

            # 1. Scavenging check
            if scavenging_rate > 0:
                if np.random.rand() < scavenging_rate:
                    states[i] = STATE_REMOVED
                    newly_removed_by_node[nb.get_thread_id(), nid] += 1
                    newly_scavenged_by_node[nb.get_thread_id(), nid] += 1
                    continue # Scavenged, so skip decomposition

            # 2. Decomposition check
            ctimers[i] -= 1
            if ctimers[i] == 0:
                states[i] = STATE_REMOVED
                daily_spores[nb.get_thread_id(), nid] += spores_per_carcass
                newly_removed_by_node[nb.get_thread_id(), nid] += 1

class CarcassDynamics:
    """
    Carcass -> Removed (Scavenged or Decomposed)
    Produces spores if decomposed.
    """
    def __init__(self, model, scavenging_rate, spores_per_carcass):
        self.model = model
        self.scavenging_rate = scavenging_rate
        self.spores_per_carcass = spores_per_carcass

        self.model.nodes.add_vector_property("newly_removed", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("newly_scavenged", model.params.nticks + 1, dtype=np.int32)

        if not hasattr(self.model.nodes, 'daily_spores'):
             self.model.nodes.add_vector_property("daily_spores", model.params.nticks + 1, dtype=np.float32)

    def step(self, tick):
        newly_removed_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        newly_scavenged_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        daily_spores_thread = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.float32)

        nb_carcass_step(
            self.model.people.state,
            self.model.people.ctimer,
            self.model.people.nodeid,
            self.scavenging_rate,
            self.spores_per_carcass,
            daily_spores_thread,
            newly_removed_by_node,
            newly_scavenged_by_node
        )

        newly_removed = newly_removed_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        newly_scavenged = newly_scavenged_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        daily_spores = daily_spores_thread.sum(axis=0)

        # Update counts
        self.model.nodes.C[tick + 1] -= newly_removed
        self.model.nodes.newly_removed[tick] = newly_removed
        self.model.nodes.newly_scavenged[tick] = newly_scavenged

        # Register new spores
        self.model.nodes.daily_spores[tick] += daily_spores # Add to any existing (though usually 0)
