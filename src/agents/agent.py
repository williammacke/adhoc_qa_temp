"""
Defines abstract base clsas of what an Agent should be like. It can get more and more complicated as time passes, with complications inside agent models.
"""


class Policy:
    def __call__(self, obs):
        raise NotImplementedError
