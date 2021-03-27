import numpy as np


class Ensemble_Model:

    def __init__(self,models,voting_type):
        self.models = models
        self.voting_type = voting_type

        voting_styles = ['average','binaverage']
        if voting_type not in voting_styles:
            raise ValueError(f'Voting style not supported: One of {voting_styles} required.')

    def vote(self,actions):
        if self.voting_type == 'average':
            return np.mean(actions,axis=0)

        elif self.voting_type == 'binaverage':
            def randargmax(b,**kw):
                """ a random tie-breaking argmax"""
                return np.argmax(np.random.random(b.shape) * (b==np.amax(b,**kw, keepdims=True)), **kw)


            # Digitize the actions according to bin
            bins = np.linspace(-1,1,6)
            digits = np.digitize(actions,bins =bins )
            counts = np.apply_along_axis(np.bincount,axis =0,arr = digits,minlength = np.max(digits) +1)

            # Get the bin with the most votes
            m = randargmax(counts,axis = 0)

            # Create mask for elements
            mask = (digits != m)

            voted_for = np.ma.masked_array(actions,mask = mask)

            # Find mean of masked array
            return np.mean(voted_for,axis = 0)

    def predict(self,state):
        ''' predict the next state using an ensemble voting pattern'''
        actions = []
        for model in self.models:
            action,_ = model.predict(state)
            actions.append(action)

        actions = np.array(actions)
        action = self.vote(actions)

        return action, None # None is a dummy for compatibility
