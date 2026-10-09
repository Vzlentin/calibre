from calibre.conformal.calibrators.base import Calibrator


class EnbPI(Calibrator):
    # def __init__(self):

    def initial_state(self, n_nodes, n_columns):
        # return {"q": np.zeros((n_nodes, n_columns))}
        pass

    def update(self, state, feedback):
        pass

    def threshold(self, state):
        # return state["q"].astype(np.float32)
        pass
