# discrete_sf2.py -- wrappers que preparan RetroSF2Env para el track Rainbow.
#
# Tres piezas apilables, todas gymnasium.Wrapper estandar:
#
#   FlatDiscreteActions   MultiDiscrete([9, 7]) -> Discrete(63) via
#                         divmod(a, 7): EXACTAMENTE la convencion del ES
#                         (es/policy.py act()) para que los tres tracks
#                         compartan mapa de acciones y el banco los compare
#                         sin traduccion.
#   StateRotation         cada reset sortea un savestate de la rotacion
#                         (uniforme, rng propio seedeado) y opcionalmente
#                         desfasa el arranque 0..K frames neutrales -- la
#                         leccion de robustez de la run 1: un agente que
#                         nunca ve la misma pelicula dos veces no puede
#                         memorizar coreografias.
#   CharOneHotObs         obs (92,) -> (212,) con es.policy.expand_char_onehot:
#                         la leccion del ciclado de matchups, aplicada aqui
#                         desde el dia uno. Sin torch: numpy puro.
#
# El orden natural: CharOneHotObs(StateRotation(FlatDiscreteActions(env))).

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium es dep dura del proyecto
    import gym
    from gym import spaces

from es.policy import ONEHOT_OBS_DIM, expand_char_onehot


class FlatDiscreteActions(gym.ActionWrapper):
    """Discrete(63) -> MultiDiscrete([9, 7]) por divmod(a, 7)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(63)

    def action(self, act):
        move, attack = divmod(int(act), 7)
        return np.array([move, attack], dtype=np.int64)


class StateRotation(gym.Wrapper):
    """Sortea el savestate por episodio; opcionalmente desfasa el arranque.

    `states` es la lista de la rotacion (p. ej. resolve_states('manifest',
    '1')). `desync_max` > 0 ejecuta 0..K acciones neutrales tras el reset
    ANTES de devolver el control; la observacion devuelta es la del ultimo
    frame neutral, asi el agente arranca donde de verdad va a jugar. Si el
    episodio terminara durante el desfase (imposible con K sensato pero
    barato de cubrir), se re-resetea sin desfase.
    """

    def __init__(self, env, states, seed=0, desync_max=0, neutral_action=0):
        super().__init__(env)
        if not states:
            raise ValueError("StateRotation requiere una lista de estados no vacia")
        self.states = list(states)
        self.desync_max = int(desync_max)
        # la accion neutral EN EL ESPACIO del env envuelto: este wrapper vive
        # encima de FlatDiscreteActions, asi que el default es el 0 plano
        # (divmod(0, 7) = (0, 0) = sin direccion, sin boton)
        self.neutral_action = neutral_action
        self._rng = np.random.default_rng(np.random.SeedSequence(
            entropy=815321, spawn_key=(int(seed),)))
        self.current_state = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        # un estado pedido explicitamente (banco, debugging) gana al sorteo
        if "state" not in options:
            options["state"] = self.states[int(self._rng.integers(len(self.states)))]
        self.current_state = options["state"]
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(int(self._rng.integers(0, self.desync_max + 1))
                       if self.desync_max else 0):
            obs, _r, term, trunc, info = self.env.step(self.neutral_action)
            if term or trunc:
                obs, info = self.env.reset(options=options)
                break
        info = dict(info)
        info["rotation_state"] = self.current_state
        return obs, info


class CharOneHotObs(gym.ObservationWrapper):
    """(92,) v4 -> (212,) con one-hot de character IDs (mismo map que el ES)."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(ONEHOT_OBS_DIM,),
                                            dtype=np.float32)

    def observation(self, obs):
        return expand_char_onehot(obs)


def make_discrete_sf2(states, seed=0, desync_max=0, onehot=True, env_kwargs=None,
                      macros=False):
    """Fabrica completa, importable desde el env_fn de un AsyncVectorEnv.

    Import tardio de RetroSF2Env: la fabrica corre DENTRO del proceso hijo
    (stable-retro solo tolera un emulador por proceso), y este modulo debe
    poder importarse en el padre sin emulador.

    macros=True: Discrete(72) via MacroActionWrapper (los 9 macros del equipo
    como opciones atomicas) en vez del Discrete(63) plano. La accion neutral
    sigue siendo 0 en ambos casos (divmod(0,7) = nada), asi que StateRotation
    no cambia. El wrapper de macros lee rel_x del frame CRUDO (indice 2 del
    frame de 23), por eso va pegado al env, debajo de todo lo demas.
    """
    from envs.retro_env import RetroSF2Env
    env = RetroSF2Env(**(env_kwargs or {}))
    if macros:
        from envs.macro_wrapper import MacroActionWrapper
        from es.policy import OBS_FRAME_DIM
        env = MacroActionWrapper(env, obs_rel_x_index=2,
                                 frame_size=OBS_FRAME_DIM)
    else:
        env = FlatDiscreteActions(env)
    env = StateRotation(env, states, seed=seed, desync_max=desync_max)
    if onehot:
        env = CharOneHotObs(env)
    return env
