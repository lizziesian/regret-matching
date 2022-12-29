from regretmatching.model import RegretMatchingDecisionMaker, Expert
import numpy as np

FEZZ = Expert('5,0,0')
FRZO = Expert('4,0,1')
FROZ = Expert('4,1,0')
THTWZ = Expert('3,2,0')
THZTW = Expert('3,0,2')
THOO = Expert('3,1,1')
TWTHZ = Expert('2,3,0')
TWZTH = Expert('2,0,3')
TWTWO = Expert('2,2,1')
TWOTW = Expert('2,1,2')
OFOZ = Expert('1,4,0')
OZFO = Expert('1,0,4')
OOTH = Expert('1,1,3')
OTHO = Expert('1,3,1')
OTWTW = Expert('1,2,2')
ZFEZ = Expert('0,5,0')
ZFOO = Expert('0,4,1')
ZTHTW = Expert('0,3,2')
ZTWTH = Expert('0,2,3')
ZOFO = Expert('0,1,4')
ZZFE = Expert('0,0,5')

BLOTTO_EXPERTS = [FEZZ, FRZO, FROZ, THTWZ, THZTW, THOO, TWTHZ, TWZTH, TWTWO, 
TWOTW, OFOZ, OZFO, OOTH, OTHO, OTWTW, ZFEZ, ZFOO, ZTHTW, ZTWTH, ZOFO, ZZFE]

BLOTTO_REWARD_VECTORS = {
    
    FEZZ:     np.asarray([0,  0,  0,  0,  0, 1,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0]),
    FRZO:     np.asarray([0,  0,  0, -1,  0, 0, -1,  0,  0,  1, -1,  0,  1,  0,  1, -1,  0,  1,  1,  1,  0]), 
    FROZ:     np.asarray([0,  0,  0,  0, -1, 0,  0, -1,  1,  0,  0, -1,  0,  1,  1,  0,  1,  1,  1,  0, -1]),
    THTWZ:    np.asarray([0,  1,  0,  0,  0, 0,  0,  1,  0, -1,  0, -1, -1,  1,  0,  0,  1,  1,  0, -1, -1]),
    THZTW:    np.asarray([0,  0,  1,  0,  0, 0, -1,  0, -1,  0, -1,  0,  1, -1,  0, -1, -1,  0,  1,  1,  0]), 
    THOO:     np.asarray([-1, 0,  0,  0,  0, 0, -1, -1,  0,  0, -1, -1,  0,  0,  1, -1,  0,  1,  1,  0, -1]),  
    TWTHZ:    np.asarray([0,  1,  0,  0,  1, 1,  0,  0,  0,  0,  0, -1, -1,  0, -1,  0,  1,  0, -1, -1, -1]),  
    TWZTH:    np.asarray([0,  0,  1, -1,  0, 1,  0,  0,  0,  0, -1,  0,  0, -1, -1, -1, -1, -1,  0,  1,  0]),  
    TWTWO:    np.asarray([-1, 0, -1,  0,  1, 0,  0,  0,  0,  0, -1, -1, -1,  0,  0, -1,  0,  1,  0, -1, -1]),  
    TWOTW:    np.asarray([-1, -1, 0,  1,  0, 0,  0,  0,  0,  0, -1, -1,  0, -1,  0, -1, -1,  0,  1,  0, -1]), 
    OFOZ:     np.asarray([0,  1,  0,  0,  1, 1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1]),  
    OZFO:     np.asarray([0,  0,  1,  1,  0, 1,  1,  0,  1,  1,  0,  0,  0,  0,  0, -1, -1, -1, -1,  0,  0]),  
    OOTH:     np.asarray([-1, -1, 0,  1, -1, 0,  1,  0,  1,  0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0, -1]),  
    OTHO:     np.asarray([-1, 0, -1, -1,  1, 0,  0,  1,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0, -1, -1, -1]), 
    OTWTW:    np.asarray([-1, -1, -1, 0,  0, -1, 1,  1,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0, -1, -1]),  
    ZFEZ:     np.asarray([0,  1,  0,  0,  1, 1,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0]),
    ZFOO:     np.asarray([-1, 0, -1, -1,  1, 0, -1,  1,  0,  1,  0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0]),  
    ZTHTW:    np.asarray([-1,-1, -1, -1,  0, -1, 0,  1, -1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0]),  
    ZTWTH:    np.asarray([-1,-1, -1,  0, -1, -1, 1,  0,  0, -1,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0]),  
    ZOFO:     np.asarray([-1, -1, 0,  1, -1, 0,  1, -1,  1,  0,  1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0]),  
    ZZFE:     np.asarray([0,  0,  1,  1,  0, 1,  1,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0]),  
}


class BLOTTOPlayer(RegretMatchingDecisionMaker):
    def __init__(self):
        super(BLOTTOPlayer, self).__init__(BLOTTO_EXPERTS)
        self.sum_p = np.full(21, 0.)
        self.games_played = 0

    def move(self):
        return self.decision()

    def learn_from(self, opponent_move):
        reward_vector = BLOTTO_REWARD_VECTORS[opponent_move]
        self.update_rule(reward_vector)
        self.games_played += 1
        self.sum_p += self.p

    def current_best_response(self):
        return np.round(self.sum_p / self.games_played, 4)

    def eps(self):
        return np.max(self.regrets / self.games_played)
